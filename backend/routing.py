from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

LatLon = Tuple[float, float]

def haversine_m(a: LatLon, b: LatLon) -> float:
    # Accurate enough for city-scale routing
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))

@dataclass
class RoadGraph:
    nodes: Dict[int, LatLon]
    adj: Dict[int, List[int]]
    edges: Dict[Tuple[int, int], Dict[str, float]]  # length_m, speed_mps, capacity

    @property
    def node_ids(self) -> List[int]:
        return list(self.nodes.keys())

    def to_json(self) -> Dict[str, Any]:
        return {
            "nodes": {str(k): [v[0], v[1]] for k, v in self.nodes.items()},
            "adj": {str(k): v for k, v in self.adj.items()},
            "edges": {f"{u}:{v}": attrs for (u, v), attrs in self.edges.items()},
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "RoadGraph":
        nodes = {int(k): (float(v[0]), float(v[1])) for k, v in d["nodes"].items()}
        adj = {int(k): [int(x) for x in v] for k, v in d["adj"].items()}
        edges = {}
        for k, attrs in d["edges"].items():
            u, v = k.split(":")
            edges[(int(u), int(v))] = {str(a): float(b) for a, b in attrs.items()}
        return RoadGraph(nodes=nodes, adj=adj, edges=edges)

    def export_for_frontend(self) -> Dict[str, Any]:
        # Returns a list of polyline segments
        segs = []
        seen = set()
        for (u, v), attrs in self.edges.items():
            if (u, v) in seen:
                continue
            seen.add((u, v))
            a = self.nodes[u]
            b = self.nodes[v]
            segs.append({"a": [a[0], a[1]], "b": [b[0], b[1]]})
        return {"segments": segs}

    def nearest_node(self, lat: float, lon: float) -> int:
        best = None
        best_d = 1e18
        for nid, (nlat, nlon) in self.nodes.items():
            d = haversine_m((lat, lon), (nlat, nlon))
            if d < best_d:
                best_d = d
                best = nid
        return int(best)

    def edge_length_m(self, u: int, v: int) -> float:
        return self.edges.get((u, v), {}).get("length_m", haversine_m(self.nodes[u], self.nodes[v]))

    def edge_speed_mps(self, u: int, v: int) -> float:
        return self.edges.get((u, v), {}).get("speed_mps", 11.11)  # ~40 km/h

    def edge_capacity(self, u: int, v: int) -> float:
        return self.edges.get((u, v), {}).get("capacity", 10.0)

    def vehicle_latlon(self, veh) -> LatLon:
        # Interpolate along current edge
        e = veh.current_edge()
        if not e:
            return self.nodes[veh.dest]
        u, v = e
        a = self.nodes[u]
        b = self.nodes[v]
        length = self.edge_length_m(u, v)
        t = 0.0 if length <= 1e-6 else max(0.0, min(1.0, veh.edge_progress_m / length))
        lat = a[0] + (b[0] - a[0]) * t
        lon = a[1] + (b[1] - a[1]) * t
        return (lat, lon)

    def route_remaining_m(self, route: List[int], edge_index: int, edge_progress_m: float) -> float:
        if len(route) < 2:
            return 0.0
        dist = 0.0
        # remaining on current edge
        if edge_index < len(route) - 1:
            u = route[edge_index]
            v = route[edge_index + 1]
            dist += max(0.0, self.edge_length_m(u, v) - edge_progress_m)
        for i in range(edge_index + 1, len(route) - 1):
            dist += self.edge_length_m(route[i], route[i + 1])
        return dist

    def shortest_path(self, start: int, goal: int, edge_cost_fn: Callable[[int, int], float]) -> List[int]:
        # Dijkstra (graph size is moderate)
        import heapq
        pq = [(0.0, start)]
        dist = {start: 0.0}
        prev: Dict[int, Optional[int]] = {start: None}
        while pq:
            d, u = heapq.heappop(pq)
            if u == goal:
                break
            if d != dist.get(u, 1e18):
                continue
            for v in self.adj.get(u, []):
                w = edge_cost_fn(u, v)
                nd = d + w
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if goal not in prev:
            return [start]
        # reconstruct
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return path

def _speed_for_highway(tag: str) -> float:
    # m/s
    # Conservative defaults
    if tag in ("motorway", "trunk"):
        return 22.2  # 80 km/h
    if tag in ("primary",):
        return 16.7  # 60 km/h
    if tag in ("secondary", "tertiary"):
        return 13.9  # 50 km/h
    if tag in ("residential", "unclassified", "service"):
        return 8.3   # 30 km/h
    return 11.1

def _capacity_for_highway(tag: str) -> float:
    if tag in ("motorway", "trunk", "primary"):
        return 30.0
    if tag in ("secondary", "tertiary"):
        return 20.0
    return 12.0

def build_graph_from_overpass_json(data: Dict[str, Any]) -> RoadGraph:
    # Parse Overpass elements: nodes + ways (highway)
    nodes: Dict[int, LatLon] = {}
    ways = []
    for el in data.get("elements", []):
        if el.get("type") == "node":
            nid = int(el["id"])
            nodes[nid] = (float(el["lat"]), float(el["lon"]))
        elif el.get("type") == "way":
            tags = el.get("tags", {})
            if "highway" in tags and "nodes" in el:
                ways.append(el)

    adj: Dict[int, List[int]] = {nid: [] for nid in nodes.keys()}
    edges: Dict[Tuple[int, int], Dict[str, float]] = {}

    def add_edge(u: int, v: int, hw: str) -> None:
        if u not in nodes or v not in nodes:
            return
        if v not in adj[u]:
            adj[u].append(v)
        length = haversine_m(nodes[u], nodes[v])
        edges[(u, v)] = {
            "length_m": length,
            "speed_mps": _speed_for_highway(hw),
            "capacity": _capacity_for_highway(hw),
        }

    for way in ways:
        hw = way.get("tags", {}).get("highway", "residential")
        oneway = way.get("tags", {}).get("oneway", "no") in ("yes", "true", "1")
        nlist = way.get("nodes", [])
        for i in range(len(nlist) - 1):
            u = int(nlist[i])
            v = int(nlist[i + 1])
            add_edge(u, v, hw)
            if not oneway:
                add_edge(v, u, hw)

    # Prune isolated nodes for performance
    used = {u for u, outs in adj.items() if outs}
    nodes = {k: v for k, v in nodes.items() if k in used}
    adj = {k: [x for x in v if x in nodes] for k, v in adj.items() if k in nodes}
    edges = {(u, v): attrs for (u, v), attrs in edges.items() if u in nodes and v in nodes}
    return RoadGraph(nodes=nodes, adj=adj, edges=edges)

def demo_graph_prishtina() -> RoadGraph:
    # A tiny fallback network centered on Prishtina (for offline runs).
    clat, clon = 42.6629, 21.1655
    step = 0.008
    nodes: Dict[int, LatLon] = {}
    nid = 1
    grid = []
    for r in range(6):
        row = []
        for c in range(6):
            lat = clat + (r - 2.5) * step
            lon = clon + (c - 2.5) * step
            nodes[nid] = (lat, lon)
            row.append(nid)
            nid += 1
        grid.append(row)
    adj: Dict[int, List[int]] = {k: [] for k in nodes}
    edges: Dict[Tuple[int, int], Dict[str, float]] = {}
    def add(u, v):
        if v not in adj[u]:
            adj[u].append(v)
        length = haversine_m(nodes[u], nodes[v])
        edges[(u, v)] = {"length_m": length, "speed_mps": 11.1, "capacity": 18.0}
    for r in range(6):
        for c in range(6):
            u = grid[r][c]
            if r + 1 < 6:
                v = grid[r+1][c]
                add(u, v); add(v, u)
            if c + 1 < 6:
                v = grid[r][c+1]
                add(u, v); add(v, u)
    return RoadGraph(nodes=nodes, adj=adj, edges=edges)
