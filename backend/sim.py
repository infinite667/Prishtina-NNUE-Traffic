from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from .routing import RoadGraph, build_graph_from_overpass_json, demo_graph_prishtina, haversine_m
from .nnue import TinyNNUE

LatLon = Tuple[float, float]

PRISHTINA_CENTER = (42.6629, 21.1655)
DEFAULT_BBOX = (42.62, 21.11, 42.71, 21.22)  # south, west, north, east

@dataclass
class SimSettings:
    nnue_enabled: bool = False
    speed_multiplier: float = 1.0
    # Fleet size target (UI slider constrains allowed values)
    target_vehicle_count: int = 500

    def model_dump(self) -> Dict[str, Any]:
        return {
            "nnue_enabled": self.nnue_enabled,
            "speed_multiplier": self.speed_multiplier,
            "target_vehicle_count": self.target_vehicle_count,
        }

    def apply_patch(self, patch: Dict[str, Any]) -> None:
        if "nnue_enabled" in patch:
            self.nnue_enabled = bool(patch["nnue_enabled"])
        if "speed_multiplier" in patch:
            val = float(patch["speed_multiplier"])
            self.speed_multiplier = max(0.5, min(5.0, val))
        if "target_vehicle_count" in patch:
            # Only allow the discrete UI values.
            try:
                v = int(patch["target_vehicle_count"])
            except Exception:
                return
            allowed = {200, 500, 1000, 3000}
            if v in allowed:
                self.target_vehicle_count = v

@dataclass
class TrafficLight:
    id: str
    lat: float
    lon: float
    cycle_s: float = 60.0
    green_s: float = 27.0
    yellow_s: float = 3.0
    offset_s: float = 0.0
    # For simplicity we treat lights as a 2-phase signal: green/yellow/red (for "its direction").
    # In real networks, direction groups are complex; here it mainly creates queuing behavior.
    def state(self, t: float) -> str:
        x = (t + self.offset_s) % self.cycle_s
        if x < self.green_s:
            return "GREEN"
        if x < self.green_s + self.yellow_s:
            return "YELLOW"
        return "RED"

@dataclass
class Vehicle:
    id: int
    color: str
    origin: int
    dest: int
    route: List[int]
    edge_index: int = 0
    pos_u: int = 0
    pos_v: int = 0
    edge_progress_m: float = 0.0
    speed_mps: float = 0.0
    distance_traveled_m: float = 0.0
    spawned_t: float = 0.0
    arrived: bool = False
    arrived_t: Optional[float] = None
    waiting_s: float = 0.0
    last_waiting: bool = False
    # After passing a GREEN light, ignore any RED-light stopping checks for a short grace period.
    # This models protected turning movements where the downstream lane's signal shouldn't block the turn.
    green_immunity_until: float = 0.0

    rerouted_by_nnue: bool = False

    def current_edge(self) -> Optional[Tuple[int, int]]:
        if self.arrived or self.edge_index >= len(self.route) - 1:
            return None
        return self.route[self.edge_index], self.route[self.edge_index + 1]

class Simulation:
    def __init__(self, data_dir: Path, settings: SimSettings):
        self.data_dir = data_dir
        self.settings = settings

        self.t0 = time.time()
        self.sim_time = 0.0

        self.graph: RoadGraph = self._load_or_fetch_graph()
        self.lights: List[TrafficLight] = self._load_lights()

        # Reverse adjacency + reachability cache used by the lightweight rerouter.
        # This prevents the AI from sending cars into dead-ends / disconnected components.
        self._rev_adj: Optional[Dict[int, List[int]]] = None
        self._reachable_cache: Dict[int, set] = {}

        # Bind each light to nearest node for queueing
        self.light_node: Dict[str, int] = {}
        for tl in self.lights:
            self.light_node[tl.id] = self.graph.nearest_node(tl.lat, tl.lon)

        # Congestion counters per directed edge
        self.edge_occupancy: Dict[Tuple[int, int], int] = {}

        self.vehicles: Dict[int, Vehicle] = {}
        self._next_vid = 1

        # Online NNUE
        self.nnue = TinyNNUE(input_dim=6, hidden_dim=32)
        self._nnue_training_accum: List[Tuple[List[float], float]] = []

        # Metrics history for UI graph
        self.avg_waiting_history: List[Tuple[float, float]] = []  # (sim_time, avg_waiting_s)
        self._last_metric_push = 0.0

        # Spawn a steady stream (spawns faster until the target fleet size is reached)
        self.spawn_rate_vps = 10.0  # vehicles per second
        self._spawn_accum = 0.0

        # Minimum bumper-to-bumper gap between queued vehicles on the same edge (meters)
        # Base gap. Actual enforced gap is adjusted per-edge to avoid short-edge pileups.
        # Patch 6.3: increase minimum gap to 10m.
        self.min_gap_m = 10.0

        # Performance safeguards: avoid spawning/removing hundreds of vehicles in a single
        # tick (which can freeze the browser due to massive marker churn).
        self.max_spawn_per_step = 35
        self.max_remove_per_reconcile = 150

    def reconcile_fleet(self) -> None:
        """Immediately reconcile the active fleet size with the current target.

        The UI expects vehicle count changes to be visible quickly after moving the slider.
        """
        target = int(self.settings.target_vehicle_count)
        if target < 0:
            target = 0

        # If we need more vehicles, spawn a SMALL batch right away so the UI reflects
        # the change quickly, but avoid large bursts that can freeze the browser.
        if len(self.vehicles) < target:
            burst = min(30, target - len(self.vehicles))
            for _ in range(burst):
                self._spawn_vehicle()
            # Give the regular spawner some extra "credit" so it catches up faster
            # without a single huge burst.
            deficit = target - len(self.vehicles)
            if deficit > 0:
                self._spawn_accum += min(250.0, float(deficit))
            # Let the normal spawn loop fill the rest over subsequent ticks.
            return

        # If we need fewer, remove oldest vehicles first (arrived, then active).
        if len(self.vehicles) > target:
            remove_n = min(self.max_remove_per_reconcile, len(self.vehicles) - target)
            arrived = [v for v in list(self.vehicles.values()) if v.arrived]
            arrived.sort(key=lambda z: z.spawned_t)
            active = [v for v in list(self.vehicles.values()) if not v.arrived]
            active.sort(key=lambda z: z.spawned_t)

            to_remove: List[int] = []
            for v in arrived:
                if len(to_remove) >= remove_n:
                    break
                to_remove.append(v.id)
            if len(to_remove) < remove_n:
                for v in active:
                    if len(to_remove) >= remove_n:
                        break
                    to_remove.append(v.id)
            for vid in to_remove:
                self.vehicles.pop(vid, None)

    def _enforce_edge_gaps(self) -> None:
        """Prevent vehicles from stacking on top of each other when queueing.

        We enforce a simple minimum gap along each directed edge by clamping the
        edge progress of following vehicles behind the vehicle in front.
        """
        by_edge: Dict[Tuple[int, int], List[Vehicle]] = {}
        for v in list(self.vehicles.values()):
            if v.arrived:
                continue
            e = v.current_edge()
            if not e:
                continue
            by_edge.setdefault(e, []).append(v)

        for (u, w), vs in by_edge.items():
            if len(vs) < 2:
                continue
            # frontmost first
            vs.sort(key=lambda z: z.edge_progress_m, reverse=True)
            # Enforce a strict bumper-to-bumper gap on the same directed edge.
            # Junction spillback handling (below) prevents overfilling short edges.
            gap = float(self.min_gap_m)
            front = vs[0]
            front_prog = front.edge_progress_m
            for follower in vs[1:]:
                max_prog = max(0.0, front_prog - gap)
                if follower.edge_progress_m > max_prog:
                    # Do not move backwards here; freeze and treat as queued.
                    follower.speed_mps = 0.0
                    follower.last_waiting = True
                front_prog = min(front_prog, follower.edge_progress_m)

    def _load_lights(self) -> List[TrafficLight]:
        geo = json.loads((self.data_dir / "traffic_lights.geojson").read_text(encoding="utf-8"))
        lights: List[TrafficLight] = []
        for i, feat in enumerate(geo.get("features", [])):
            geom = feat.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            lights.append(TrafficLight(
                id=str(feat.get("id") or f"tl_{i}"),
                lat=lat,
                lon=lon,
                offset_s=random.random() * 60.0,
            ))
        return lights

    def _load_or_fetch_graph(self) -> RoadGraph:
        cache_path = self.data_dir / "roads_cache.json"
        if cache_path.exists():
            try:
                return RoadGraph.from_json(json.loads(cache_path.read_text(encoding="utf-8")))
            except Exception:
                pass

        # Try Overpass
        try:
            overpass_json = self._fetch_overpass_roads()
            graph = build_graph_from_overpass_json(overpass_json)
            cache_path.write_text(json.dumps(graph.to_json()), encoding="utf-8")
            return graph
        except Exception:
            # Fallback demo network (still centered on Prishtina)
            return demo_graph_prishtina()

    def _fetch_overpass_roads(self) -> Dict[str, Any]:
        south, west, north, east = DEFAULT_BBOX
        # Avoid osmnx; use a direct Overpass query.
        query = f"""
        [out:json][timeout:25];
        (
          way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"]({south},{west},{north},{east});
        );
        (._;>;);
        out body;
        """
        url = "https://overpass-api.de/api/interpreter"
        resp = requests.post(url, data={"data": query}, timeout=45)
        resp.raise_for_status()
        return resp.json()

    def road_export(self) -> Dict[str, Any]:
        return self.graph.export_for_frontend()

    def lights_export(self) -> List[Dict[str, Any]]:
        out = []
        for tl in self.lights:
            out.append({
                "id": tl.id,
                "lat": tl.lat,
                "lon": tl.lon,
                "state": tl.state(self.sim_time),
                "node": self.light_node.get(tl.id),
            })
        return out

    def metrics_export(self) -> Dict[str, Any]:
        avg_wait = self._avg_waiting_s()
        # sim_time is advanced using wall-clock dt in backend/main.py, so it's already in seconds.
        return {"sim_time": self.sim_time, "avg_waiting_s": round(avg_wait, 1)}

    def snapshot_export(self) -> Dict[str, Any]:
        return {
            "t": self.sim_time,
            "vehicles": [self._vehicle_payload(v) for v in list(self.vehicles.values())],
            "lights": self.lights_export(),
            "metrics": self.metrics_export(),
        }

    def _vehicle_payload(self, v: Vehicle) -> Dict[str, Any]:
        lat, lon = self.graph.vehicle_latlon(v)
        return {
            "id": v.id,
            "lat": lat,
            "lon": lon,
            "state": "ARRIVED" if v.arrived else ("WAITING" if v.last_waiting else "MOVING"),
            "speed_kmh": round(v.speed_mps * 3.6, 1),
            "distance_traveled_m": round(v.distance_traveled_m, 1),
            "distance_to_dest_m": round(self.graph.route_remaining_m(v.route, v.edge_index, v.edge_progress_m), 1),
            "time_in_sim_s": round(self.sim_time - v.spawned_t, 1),
            "waiting_time_s": round(v.waiting_s, 1),
            "rerouted": v.rerouted_by_nnue,
        }

    def _avg_waiting_s(self) -> float:
        # Average accumulated waiting for ACTIVE (non-arrived) vehicles.
        active = [v for v in list(self.vehicles.values()) if not v.arrived]
        if not active:
            return 0.0
        return sum(v.waiting_s for v in active) / len(active)

    def _spawn_vehicle(self) -> None:
        # Try a few times to find a valid route and a clear start edge to avoid spawning overlaps.
        gap = float(self.min_gap_m)
        for _ in range(20):
            origin = random.choice(self.graph.node_ids)
            dest = random.choice(self.graph.node_ids)
            while dest == origin:
                dest = random.choice(self.graph.node_ids)

            route = self.graph.shortest_path(origin, dest, edge_cost_fn=self._edge_cost_base)
            if len(route) < 2:
                continue

            start_edge = (route[0], route[1])
            backmost = None
            for vv in list(self.vehicles.values()):
                if vv.arrived:
                    continue
                ee = vv.current_edge()
                if ee == start_edge:
                    backmost = vv.edge_progress_m if backmost is None else min(backmost, vv.edge_progress_m)
            if backmost is not None and backmost < gap:
                continue

            v = Vehicle(
                id=self._next_vid,
                color="blue",
                origin=origin,
                dest=dest,
                route=route,
                pos_u=route[0],
                pos_v=route[1],
                spawned_t=self.sim_time,
            )
            self.vehicles[v.id] = v
            self._next_vid += 1
            return

    def _respawn_vehicle(self, v: Vehicle) -> None:
        """Respawn an arrived vehicle to a new random origin/destination."""
        # Try a few times to find a valid route and a clear start edge.
        for _ in range(20):
            origin = random.choice(self.graph.node_ids)
            dest = random.choice(self.graph.node_ids)
            while dest == origin:
                dest = random.choice(self.graph.node_ids)

            route = self.graph.shortest_path(origin, dest, edge_cost_fn=self._edge_cost_base)
            if len(route) < 2:
                continue

            e = (route[0], route[1])
            # Require clearance at the start of the edge (spillback / min-gap)
            gap = float(self.min_gap_m)
            backmost = None
            for vv in list(self.vehicles.values()):
                if vv.id == v.id or vv.arrived:
                    continue
                ee = vv.current_edge()
                if ee == e:
                    backmost = vv.edge_progress_m if backmost is None else min(backmost, vv.edge_progress_m)
            if backmost is not None and backmost < gap:
                continue

            # Reset vehicle state
            v.origin = origin
            v.dest = dest
            v.route = route
            v.edge_index = 0
            v.pos_u = route[0]
            v.pos_v = route[1]
            v.edge_progress_m = 0.0
            v.speed_mps = 0.0
            v.distance_traveled_m = 0.0
            v.spawned_t = self.sim_time
            v.arrived = False
            v.arrived_t = None
            v.waiting_s = 0.0
            v.last_waiting = False
            v.rerouted_by_nnue = False
            v.green_immunity_until = 0.0
            v.color = "blue"
            return

        # If we couldn't respawn (very congested), just keep it arrived a bit longer.
        v.arrived_t = self.sim_time

    def _edge_cost_base(self, u: int, v: int) -> float:
        # Base travel time estimate (seconds)
        length_m = self.graph.edge_length_m(u, v)
        speed_mps = self.graph.edge_speed_mps(u, v)
        return length_m / max(1.0, speed_mps)

    def _edge_cost_nnue(self, u: int, v: int) -> float:
        base = self._edge_cost_base(u, v)
        # Features: base_time, occupancy, capacity, avg_waiting, light_red_prob, sim_speed
        occ = float(self.edge_occupancy.get((u, v), 0))
        cap = float(self.graph.edge_capacity(u, v))
        avgw = float(self._avg_waiting_s())
        red = 0.0
        # if there is a light near v and it's red, add a signal feature
        for tl in self.lights:
            if self.light_node.get(tl.id) == v:
                red = 1.0 if tl.state(self.sim_time) == "RED" else 0.0
                break
        x = [base, occ, cap, avgw, red, float(self.settings.speed_multiplier)]
        mult = self.nnue.predict_multiplier(x)  # ~1.0+
        # Always positive
        return base * max(0.6, min(3.0, mult))

    def _heuristic_time_to_dest(self, node: int, dest: int) -> float:
        """A cheap optimistic estimate (seconds) from node -> dest."""
        try:
            a = self.graph.nodes[node]
            b = self.graph.nodes[dest]
            # optimistic: 60 km/h
            return haversine_m(a, b) / 16.7
        except Exception:
            return 0.0

    def _get_rev_adj(self) -> Dict[int, List[int]]:
        """Build (or return cached) reverse adjacency: v -> [u] where u->v exists."""
        if self._rev_adj is not None:
            return self._rev_adj
        rev: Dict[int, List[int]] = {nid: [] for nid in self.graph.nodes.keys()}
        for u, outs in self.graph.adj.items():
            for v in outs:
                if v in rev:
                    rev[v].append(u)
        self._rev_adj = rev
        return rev

    def _reachable_to_dest(self, dest: int) -> set:
        """Nodes that can reach dest in the directed graph (computed by reverse BFS from dest).

        Used to prevent rerouting into dead ends / disconnected subgraphs.
        """
        if dest in self._reachable_cache:
            return self._reachable_cache[dest]

        # Keep the cache bounded so long runs don't grow unbounded.
        if len(self._reachable_cache) > 128:
            self._reachable_cache.clear()

        rev = self._get_rev_adj()
        seen = set()
        stack = [dest]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            for p in rev.get(x, []):
                if p not in seen:
                    stack.append(p)

        self._reachable_cache[dest] = seen
        return seen

    def _ai_build_route_greedy(self, start: int, dest: int, max_hops: int = 24) -> List[int]:
        """Lightweight rerouting that won't freeze the sim.

        Instead of running Dijkstra per vehicle (too expensive for 1000-3000 cars),
        we do a greedy lookahead: at each node choose the neighbor that minimizes
        (edge_cost + heuristic_to_dest). This is fast and still shifts flow away
        from congested / red-light nodes.
        """
        if start == dest:
            return [start]

        # Safety: only consider moves that stay within the set of nodes that can
        # actually reach the destination. This avoids NNUE sending cars into dead-ends.
        try:
            reachable = self._reachable_to_dest(dest)
            if start not in reachable:
                # start cannot reach dest in directed graph; don't attempt a reroute.
                return [start]
        except Exception:
            reachable = None
        route = [start]
        visited = {start}
        cur = start
        for _ in range(max_hops):
            if cur == dest:
                break
            nbrs = self.graph.adj.get(cur, [])
            if not nbrs:
                break

            best_v = None
            best_score = 1e18
            # Evaluate a small set of candidates (all outgoing edges from cur)
            for v in nbrs:
                if reachable is not None and v not in reachable:
                    continue
                # Avoid tight loops; still allow revisits if we're stuck.
                loop_penalty = 25.0 if v in visited else 0.0
                base_cost = self._edge_cost_base(cur, v)
                occ = float(self.edge_occupancy.get((cur, v), 0))
                cap = float(self.graph.edge_capacity(cur, v))
                congestion = min(1.5, occ / max(1.0, cap))

                # Prefer edges with spare capacity. Quadratic term makes heavy congestion expensive.
                cost = base_cost * (1.0 + 2.2 * (congestion ** 2))

                # Light penalty at the *end* node.
                red = 0.0
                for tl in self.lights:
                    if self.light_node.get(tl.id) == v:
                        red = 1.0 if tl.state(self.sim_time) == "RED" else 0.0
                        break
                cost += 6.0 * red

                # Optional NNUE multiplier (kept bounded, and cheap because it's per-neighbor).
                if self.settings.nnue_enabled:
                    try:
                        x = [base_cost, occ, cap, float(self._avg_waiting_s()), red, float(self.settings.speed_multiplier)]
                        mult = self.nnue.predict_multiplier(x)
                        cost *= max(0.7, min(2.2, mult))
                    except Exception:
                        # If NNUE ever misbehaves, ignore it (never freeze).
                        pass

                score = cost + self._heuristic_time_to_dest(v, dest) + loop_penalty
                if score < best_score:
                    best_score = score
                    best_v = v

            if best_v is None:
                break
            route.append(int(best_v))
            visited.add(int(best_v))
            cur = int(best_v)
            if cur == dest:
                break

        return route

    def _ai_build_route_astar_limited(self, start: int, dest: int, max_expansions: int = 1800) -> List[int]:
        """A* with a strict expansion budget to avoid freezes.

        Used as a fallback when greedy routing can't find a full path to dest.
        Costs include congestion + red-light penalties, but this is still bounded.
        """
        import heapq

        if start == dest:
            return [start]

        reachable = self._reachable_to(dest)
        if reachable is not None and start not in reachable:
            return [start]

        # (f, g, node)
        pq: List[Tuple[float, float, int]] = []
        g: Dict[int, float] = {start: 0.0}
        prev: Dict[int, Optional[int]] = {start: None}

        def h(n: int) -> float:
            # optimistic: straight-line distance / ~13.9 mps (50 km/h) => seconds-ish
            a = self.graph.nodes.get(n)
            b = self.graph.nodes.get(dest)
            if not a or not b:
                return 0.0
            return haversine_m(a, b) / 13.9

        heapq.heappush(pq, (h(start), 0.0, start))
        expansions = 0

        while pq and expansions < max_expansions:
            f, du, u = heapq.heappop(pq)
            if u == dest:
                break
            if du != g.get(u, 1e18):
                continue
            expansions += 1

            for v in self.graph.adj.get(u, []):
                if reachable is not None and v not in reachable:
                    continue

                base_cost = self._edge_cost_base(u, v)
                occ = float(self.edge_occupancy.get((u, v), 0))
                cap = float(self.graph.edge_capacity(u, v))
                congestion = min(1.5, occ / max(1.0, cap))
                cost = base_cost * (1.0 + 0.9 * congestion)

                # Penalize arriving into a node with a red light.
                red = 0.0
                for tl in self.lights:
                    if self.light_node.get(tl.id) == v:
                        red = 1.0 if tl.state(self.sim_time) == "RED" else 0.0
                        break
                cost += 6.0 * red

                nd = du + cost
                if nd < g.get(v, 1e18):
                    g[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd + h(v), nd, v))

        if dest not in prev:
            return [start]

        # reconstruct
        path: List[int] = []
        cur: Optional[int] = dest
        while cur is not None:
            path.append(int(cur))
            cur = prev.get(cur)
        path.reverse()
        return path

    def step(self, dt: float) -> None:
        dt *= self.settings.speed_multiplier
        self.sim_time += dt

        # Patch 6.4: keep arrived vehicles visible for a short time, then respawn them.
        for v in list(self.vehicles.values()):
            if not v.arrived:
                continue
            # If arrived_t is missing (older save), set it now.
            if v.arrived_t is None:
                v.arrived_t = self.sim_time
            if (self.sim_time - v.arrived_t) >= 5.0:
                self._respawn_vehicle(v)

        # Spawn vehicles
        self._spawn_accum += dt * self.spawn_rate_vps
        spawned_this_step = 0
        while (
            self._spawn_accum >= 1.0
            and len(self.vehicles) < self.settings.target_vehicle_count
            and spawned_this_step < int(self.max_spawn_per_step)
        ):
            self._spawn_vehicle()
            self._spawn_accum -= 1.0
            spawned_this_step += 1

        # Reset occupancy
        self.edge_occupancy = {}
        for v in list(self.vehicles.values()):
            e = v.current_edge()
            if e:
                self.edge_occupancy[e] = self.edge_occupancy.get(e, 0) + 1

        # NOTE: We intentionally DO NOT run full shortest-path routing per vehicle here.
        # Doing Dijkstra thousands of times per second will freeze the server and UI.
        # Rerouting is handled cheaply at intersections during edge transfers.

        # Move vehicles (with simple car-following / gap enforcement)
        # Key idea: enforce spacing per-edge, but adapt the gap for very short edges
        # so we don't end up clamping multiple cars to progress=0 (visual stacking).
        BASE_GAP_M = float(self.min_gap_m)
        EPS_MOVE = 2e-2  # ~2cm per tick counts as "not moving"

        # Group vehicles by edge for ordering (front -> back)
        edge_to_vs: Dict[Tuple[int, int], List[Vehicle]] = {}
        for v in list(self.vehicles.values()):
            if v.arrived:
                v.last_waiting = False
                continue
            e = v.current_edge()
            if not e:
                # If a route ended before reaching the real destination (e.g., dead-end),
                # try a bounded recovery route from the current node.
                cur_node = int(v.route[-1]) if v.route else int(v.pos_v or v.pos_u)
                if cur_node != int(v.dest):
                    try:
                        rec = self._ai_build_route_astar_limited(cur_node, int(v.dest), max_expansions=2600)
                        if len(rec) >= 2 and rec[-1] == int(v.dest):
                            v.route = rec
                            v.edge_index = 0
                            v.pos_u = rec[0]
                            v.pos_v = rec[1]
                            v.edge_progress_m = 0.0
                            v.rerouted_by_nnue = True
                            # Re-evaluate this vehicle on its new edge on next tick.
                            continue
                    except Exception:
                        pass

                # Otherwise treat it as arrived (and respawn later).
                v.arrived = True
                v.color = "purple"
                v.last_waiting = False
                v.arrived_t = self.sim_time
                continue
            edge_to_vs.setdefault(e, []).append(v)

        # Update vehicles edge-by-edge so followers can respect the updated leader position
        for (u, w), vs in edge_to_vs.items():
            # Frontmost first
            vs.sort(key=lambda z: z.edge_progress_m, reverse=True)

            length_m = self.graph.edge_length_m(u, w)
            free_speed = self.graph.edge_speed_mps(u, w)

            # Dynamic stop-line buffer and gap.
            # - buffer: keep cars a little before the node, but never larger than 20% of the edge.
            stop_line_buffer_m = min(6.0, max(1.0, 0.20 * length_m))
            # Strict rule: keep a fixed bumper-to-bumper gap even when stopped.
            gap_m = BASE_GAP_M

            occ = self.edge_occupancy.get((u, w), 0)
            cap = self.graph.edge_capacity(u, w)
            congestion = min(0.95, occ / max(1.0, cap))
            target_speed = max(0.5, free_speed * (1.0 - 0.75 * congestion))

            leader_progress = None
            leader_waiting = False

            for idx, v in enumerate(vs):
                # Traffic light gating at the end node (w)
                must_stop_light = False
                for tl in self.lights:
                    if self.light_node.get(tl.id) == w:
                        st = tl.state(self.sim_time)
                        if st == "RED":
                            # If the vehicle has recently passed a GREEN light, it has a short grace period
                            # where it ignores downstream RED checks (still respecting spacing/spillback).
                            if self.sim_time < getattr(v, "green_immunity_until", 0.0):
                                must_stop_light = False
                            else:
                                dist_to_end = max(0.0, length_m - v.edge_progress_m)
                                # start braking earlier so queues form more naturally
                                if dist_to_end < 25.0:
                                    must_stop_light = True
                        break

                desired_progress = v.edge_progress_m

                stop_at = max(0.0, length_m - stop_line_buffer_m)
                if must_stop_light:
                    # Patch 6.3: if a light turns red after we've advanced close to the stop line,
                    # never "snap" the vehicle backwards (that causes visible jitter/loops).
                    # Instead, freeze in place once we've reached the stop line.
                    if v.edge_progress_m >= stop_at:
                        desired_progress = v.edge_progress_m
                    else:
                        desired_progress = min(v.edge_progress_m + target_speed * dt, stop_at)
                else:
                    desired_progress = v.edge_progress_m + target_speed * dt

                # Enforce gap to leader on same edge
                if leader_progress is not None:
                    max_follow = leader_progress - gap_m

                    # Never move vehicles backwards (backwards snaps are the main source of visible jitter).
                    # If a vehicle is already too close (rare safety case), freeze it in place and treat it as queued.
                    if v.edge_progress_m > max_follow:
                        desired_progress = v.edge_progress_m
                    else:
                        if desired_progress > max_follow:
                            desired_progress = max_follow

                prev_progress = v.edge_progress_m
                v.edge_progress_m = max(0.0, desired_progress)

                actual_ds = v.edge_progress_m - prev_progress
                if actual_ds > 0:
                    v.distance_traveled_m += actual_ds
                    v.speed_mps = (actual_ds / dt) if dt > 1e-9 else 0.0
                else:
                    v.speed_mps = 0.0

                blocked_by_leader = (leader_progress is not None) and (leader_progress - v.edge_progress_m) <= (gap_m + 0.05)
                # Count waiting if effectively not moving AND either the light is forcing a stop,
                # or we're blocked by a (possibly stopped) leader.
                is_waiting = (actual_ds <= EPS_MOVE) and (must_stop_light or blocked_by_leader or leader_waiting)

                if is_waiting:
                    v.waiting_s += dt
                    v.last_waiting = True
                    v.color = "pink"
                else:
                    v.last_waiting = False
                    v.color = "blue"

                # update leader for next follower
                leader_progress = v.edge_progress_m
                # propagate "waiting" more robustly (prevents wait-time from dropping out mid-queue)
                leader_waiting = bool(v.last_waiting or blocked_by_leader or must_stop_light)

        # Edge completion / advancing to next edge (after movement)
        # IMPORTANT: to avoid vehicles "teleporting" onto the same node and stacking,
        # we only allow a vehicle to enter the next edge if there is at least min_gap_m
        # of free space from the start of that next edge (spillback behavior).

        gap_m = float(self.min_gap_m)

        # Precompute which vehicles are on which edge (after movement) so we can
        # query start-of-edge clearance during transfers.
        edge_to_progresses: Dict[Tuple[int, int], List[float]] = {}
        for vv in list(self.vehicles.values()):
            if vv.arrived:
                continue
            ee = vv.current_edge()
            if not ee:
                continue
            edge_to_progresses.setdefault(ee, []).append(float(vv.edge_progress_m))

        # Collect finishers per edge so we can advance them front-to-back.
        finishers: Dict[Tuple[int, int], List[Vehicle]] = {}
        for v in list(self.vehicles.values()):
            if v.arrived:
                continue
            e = v.current_edge()
            if not e:
                continue
            u, w = e
            length_m = self.graph.edge_length_m(u, w)
            if v.edge_progress_m >= length_m - 1e-6:
                finishers.setdefault((u, w), []).append(v)

        # Process per-edge so a leading vehicle transfers first.
        for (u, w), vs in finishers.items():
            vs.sort(key=lambda z: z.edge_progress_m, reverse=True)
            length_m = self.graph.edge_length_m(u, w)
            stop_line_buffer_m = min(6.0, max(1.0, 0.20 * length_m))

            # Light gating at the end node (w)
            light_state: Optional[str] = None
            light_is_red = False
            for tl in self.lights:
                if self.light_node.get(tl.id) == w:
                    light_state = tl.state(self.sim_time)
                    light_is_red = (light_state == "RED")
                    break

            for v in vs:
                if v.arrived:
                    continue

                # If red, do not transfer. Never snap backwards; just stop where you are (clamped to edge end).
                if light_is_red and (self.sim_time >= float(getattr(v, "green_immunity_until", 0.0))):
                    v.edge_progress_m = min(float(v.edge_progress_m), float(length_m))
                    v.speed_mps = 0.0
                    v.last_waiting = True
                    v.color = "pink"
                    v.waiting_s += dt
                    continue

                # If we passed this intersection on GREEN, grant a short grace period where
                # this vehicle ignores downstream RED-light stopping checks (useful for turns).
                if light_state == "GREEN":
                    v.green_immunity_until = max(float(getattr(v, "green_immunity_until", 0.0)), float(self.sim_time + 5.0))

                overshoot = float(v.edge_progress_m - length_m)
                next_index = v.edge_index + 1

                # Arrive if this was the final edge.
                if next_index >= len(v.route) - 1:
                    v.arrived = True
                    v.color = "purple"
                    v.speed_mps = 0.0
                    v.edge_progress_m = length_m
                    v.arrived_t = self.sim_time
                    continue

                # Lightweight AI reroute at intersections (fast; avoids per-vehicle Dijkstra).
                # We only consider rerouting when NNUE is enabled and either:
                # - the planned next edge looks congested, or
                # - we randomly sample a small fraction of vehicles.
                if self.settings.nnue_enabled:
                    try:
                        cur_node = int(v.route[next_index])  # this is the intersection node we're entering
                        # Planned next edge congestion signal
                        planned_v = int(v.route[next_index + 1])
                        occ = float(self.edge_occupancy.get((cur_node, planned_v), 0))
                        cap = float(self.graph.edge_capacity(cur_node, planned_v))
                        cong = occ / max(1.0, cap)
                        should_try = (cong >= 0.75) or (random.random() < 0.06)
                        if should_try:
                            dest_node = int(v.dest)
                            new_route = self._ai_build_route_greedy(cur_node, dest_node, max_hops=48)

                            # Greedy can get trapped; only accept full routes.
                            if (len(new_route) < 2) or (new_route[-1] != dest_node):
                                new_route = self._ai_build_route_astar_limited(cur_node, dest_node, max_expansions=2200)

                            if len(new_route) >= 2 and new_route[-1] == dest_node:
                                prefix = v.route[: next_index + 1]
                                v.route = prefix + new_route[1:]
                                v.rerouted_by_nnue = True
                    except Exception:
                        # Never let rerouting freeze the sim.
                        pass

                nu = v.route[next_index]
                nv = v.route[next_index + 1]
                next_edge = (nu, nv)

                # Check clearance at start of next edge.
                next_len = self.graph.edge_length_m(nu, nv)
                prog_list = edge_to_progresses.get(next_edge, [])
                if prog_list:
                    backmost = min(prog_list)
                    # If the backmost car is too close to the start, we can't enter yet.
                    if backmost < gap_m:
                        # Spillback: cannot enter next edge yet. Never snap backwards; hold position at the edge end.
                        v.edge_progress_m = min(float(v.edge_progress_m), float(length_m))
                        v.speed_mps = 0.0
                        v.last_waiting = True
                        v.color = "pink"
                        v.waiting_s += dt
                        continue
                    # Place this car behind the backmost while preserving the gap.
                    new_prog = max(0.0, min(max(0.0, overshoot), backmost - gap_m))
                else:
                    new_prog = max(0.0, overshoot)

                # If the next edge is extremely short, enforce that only one car can occupy it.
                # (Otherwise multiple cars would clamp to progress=0 and visually overlap.)
                if prog_list and next_len < gap_m:
                    # Prevent overfilling extremely short edges. Hold at the edge end without snapping backwards.
                    v.edge_progress_m = min(float(v.edge_progress_m), float(length_m))
                    v.speed_mps = 0.0
                    v.last_waiting = True
                    v.color = "pink"
                    v.waiting_s += dt
                    continue

                # Apply transfer.
                v.edge_index = next_index
                v.edge_progress_m = new_prog
                v.pos_u = nu
                v.pos_v = nv

                # Update our cache so additional finishers in the same tick see the new car.
                edge_to_progresses.setdefault(next_edge, []).append(float(v.edge_progress_m))
        # Final cleanup: enforce gaps again after edge transitions.
        # This helps avoid rare overlaps caused by overshoot/advance.
        # Metrics history (1Hz)
        if self.sim_time - self._last_metric_push >= 1.0:
            self._last_metric_push = self.sim_time
            self.avg_waiting_history.append((self.sim_time, self._avg_waiting_s()))
            # keep last 3 minutes
            self.avg_waiting_history = [p for p in self.avg_waiting_history if self.sim_time - p[0] <= 180.0]

        # Keep vehicle count bounded for performance
        # (This list must always exist even if we're below the cap.)
        to_remove: List[int] = []
        hard_cap = max(650, int(self.settings.target_vehicle_count * 1.35))
        if len(self.vehicles) > hard_cap:
            # remove oldest arrived first
            arrived = [v for v in list(self.vehicles.values()) if v.arrived]
            arrived.sort(key=lambda z: z.spawned_t)
            for v in arrived[: max(200, int(0.15 * len(self.vehicles)) )]:
                to_remove.append(v.id)
            # If we're still above cap, remove oldest non-arrived as a last resort.
            if len(self.vehicles) - len(to_remove) > hard_cap:
                active = [v for v in list(self.vehicles.values()) if not v.arrived]
                active.sort(key=lambda z: z.spawned_t)
                overflow = (len(self.vehicles) - len(to_remove)) - hard_cap
                for v in active[:overflow]:
                    to_remove.append(v.id)
        for vid in to_remove:
            self.vehicles.pop(vid, None)

    def stream_updates(self) -> Iterable[str]:
        # Async generator implemented via polling from websocket handler.
        # Yield updates as JSON strings.
        # Patch 6.3: dynamically lower the send rate for larger fleets to prevent
        # UI freezes when the vehicle count slider increases.
        async def _gen():
            last = 0.0
            while True:
                n = len(self.vehicles)
                interval = 1.0 / 30.0
                if n >= 900:
                    interval = 1.0 / 15.0
                if n >= 1800:
                    interval = 1.0 / 8.0

                # push at target interval
                if self.sim_time - last >= interval:
                    last = self.sim_time
                    payload = {
                        "type": "tick",
                        "payload": {
                            "t": self.sim_time,
                            "vehicles": [self._vehicle_payload(v) for v in list(self.vehicles.values())],
                            "lights": self.lights_export(),
                            "metrics": self.metrics_export(),
                        },
                    }
                    yield json.dumps(payload)
                await asyncio.sleep(0.001)
        import asyncio
        return _gen()