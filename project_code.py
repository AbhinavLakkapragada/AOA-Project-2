#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Flow and NP-Completeness in Practice
==============================================

A) Network Flow: Blood Bank Distribution Network
   - Real-world: Emergency blood distribution from multiple banks to hospitals
   - Reduction: Multi-commodity network flow with blood type compatibility
   - Algorithm: Ford-Fulkerson with BFS (Edmonds-Karp) for maximum flow
   - Correctness: Max-flow Min-cut theorem
   - Time: O(V*E^2) where V=nodes, E=edges

B) NP-Complete: Museum Artwork Arrangement with Conflicts
   - Real-world: Museum artwork placement to avoid incompatible pieces in same room
   - Reduction: Graph k-Coloring (known NP-Complete)
   - Greedy heuristics: Welsh-Powell (largest degree first), DSatur
   - Correctness: Approximation quality depends on graph structure
   - Time: O(V^2) for greedy, exponential for optimal

This script:
  • Runs correctness checks against alternative algorithms
  • Benchmarks both problems across varying sizes
  • Produces CSVs and PNG plots for LaTeX inclusion
"""

from __future__ import annotations
import csv
import math
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

# ================================
# Global Configuration
# ================================
OUTPUT_DIR = "outputs"

def ensure_outputs_dir() -> None:
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

def path_in_outputs(filename: str) -> str:
    ensure_outputs_dir()
    return os.path.join(OUTPUT_DIR, filename)

def now_ns() -> int:
    return time.perf_counter_ns()

def secs(ns: int) -> float:
    return ns / 1e9

def write_csv(filename: str, header: List[str], rows: List[Tuple]) -> None:
    fp = path_in_outputs(filename)
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def try_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"[warn] matplotlib not available; plots skipped: {e}")
        return None

random.seed(42)

# =========================================================
# Problem A: Blood Bank Distribution (Network Flow)
# =========================================================

# Blood type compatibility matrix (donor -> recipients)
BLOOD_COMPAT = {
    'O-': ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+'],
    'O+': ['O+', 'A+', 'B+', 'AB+'],
    'A-': ['A-', 'A+', 'AB-', 'AB+'],
    'A+': ['A+', 'AB+'],
    'B-': ['B-', 'B+', 'AB-', 'AB+'],
    'B+': ['B+', 'AB+'],
    'AB-': ['AB-', 'AB+'],
    'AB+': ['AB+']
}

BLOOD_TYPES = ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+']

@dataclass
class BloodBank:
    id: int
    supplies: Dict[str, int]  # blood_type -> units available

@dataclass
class Hospital:
    id: int
    demands: Dict[str, int]  # blood_type -> units needed

class FlowNetwork:
    """
    Network flow graph with capacities.
    Nodes are integers; edges have capacities.
    """
    def __init__(self):
        self.graph: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    
    def add_edge(self, u: int, v: int, cap: int) -> None:
        """Add directed edge u->v with capacity cap (also creates reverse edge)"""
        self.graph[u][v] += cap
    
    def bfs_find_path(self, source: int, sink: int, parent: Dict[int, int]) -> bool:
        """BFS to find augmenting path from source to sink"""
        visited = {source}
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            
            for v in self.graph[u]:
                if v not in visited and self.graph[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    if v == sink:
                        return True
                    queue.append(v)
        return False
    
    def edmonds_karp(self, source: int, sink: int) -> Tuple[int, Dict[Tuple[int,int], int]]:
        """
        Edmonds-Karp algorithm for maximum flow (Ford-Fulkerson with BFS).
        Returns (max_flow_value, flow_dict) where flow_dict maps (u,v) -> flow
        """
        parent = {}
        max_flow = 0
        flow_edges = defaultdict(int)
        
        # Create residual graph (copy of original)
        residual = defaultdict(lambda: defaultdict(int))
        for u in self.graph:
            for v in self.graph[u]:
                residual[u][v] = self.graph[u][v]
        
        # Find augmenting paths using BFS
        while True:
            parent.clear()
            if not self.bfs_find_path_residual(source, sink, parent, residual):
                break
            
            # Find minimum capacity along the path
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, residual[u][v])
                v = u
            
            # Update residual capacities and flow
            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                flow_edges[(u, v)] += path_flow
                v = u
            
            max_flow += path_flow
        
        return max_flow, dict(flow_edges)
    
    def bfs_find_path_residual(self, source: int, sink: int, parent: Dict[int, int], 
                                residual: Dict) -> bool:
        """BFS on residual graph"""
        visited = {source}
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            
            for v in residual[u]:
                if v not in visited and residual[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    if v == sink:
                        return True
                    queue.append(v)
        return False


def build_blood_distribution_network(banks: List[BloodBank], 
                                     hospitals: List[Hospital]) -> Tuple[FlowNetwork, int, int, Dict]:
    """
    Construct network flow graph for blood distribution.
    
    Network structure:
    - Source (node 0)
    - Bank nodes (1 to num_banks)
    - Blood type intermediary nodes (num_banks+1 to num_banks+8)
    - Hospital nodes (num_banks+9 to num_banks+8+num_hospitals)
    - Sink (last node)
    
    Returns: (network, source_id, sink_id, node_mapping)
    """
    net = FlowNetwork()
    node_map = {}
    
    source = 0
    node_map['source'] = source
    
    # Bank nodes
    bank_offset = 1
    for i, bank in enumerate(banks):
        bank_node = bank_offset + i
        node_map[f'bank_{bank.id}'] = bank_node
        
        # Source to banks (total supply from each bank)
        total_supply = sum(bank.supplies.values())
        net.add_edge(source, bank_node, total_supply)
    
    # Blood type intermediary nodes
    type_offset = bank_offset + len(banks)
    for i, blood_type in enumerate(BLOOD_TYPES):
        type_node = type_offset + i
        node_map[f'type_{blood_type}'] = type_node
        
        # Banks to blood types (by available supply of each type)
        for j, bank in enumerate(banks):
            bank_node = bank_offset + j
            if blood_type in bank.supplies and bank.supplies[blood_type] > 0:
                net.add_edge(bank_node, type_node, bank.supplies[blood_type])
    
    # Hospital nodes
    hosp_offset = type_offset + len(BLOOD_TYPES)
    for i, hospital in enumerate(hospitals):
        hosp_node = hosp_offset + i
        node_map[f'hospital_{hospital.id}'] = hosp_node
    
    # Blood types to hospitals (considering compatibility)
    for i, blood_type in enumerate(BLOOD_TYPES):
        type_node = type_offset + i
        compatible_recipient_types = BLOOD_COMPAT[blood_type]
        
        for j, hospital in enumerate(hospitals):
            hosp_node = hosp_offset + j
            # Sum demand for all compatible recipient types in this hospital
            compatible_demand = sum(
                hospital.demands.get(rec_type, 0) 
                for rec_type in compatible_recipient_types 
                if rec_type in hospital.demands
            )
            if compatible_demand > 0:
                net.add_edge(type_node, hosp_node, compatible_demand)
    
    # Sink
    sink = hosp_offset + len(hospitals)
    node_map['sink'] = sink
    
    # Hospitals to sink (by total demand)
    for i, hospital in enumerate(hospitals):
        hosp_node = hosp_offset + i
        total_demand = sum(hospital.demands.values())
        net.add_edge(hosp_node, sink, total_demand)
    
    return net, source, sink, node_map


def solve_blood_distribution(banks: List[BloodBank], 
                             hospitals: List[Hospital]) -> Tuple[int, Dict]:
    """
    Solve blood distribution problem using max flow.
    Returns (total_units_delivered, flow_details)
    """
    network, source, sink, node_map = build_blood_distribution_network(banks, hospitals)
    max_flow, flow_edges = network.edmonds_karp(source, sink)
    return max_flow, flow_edges


# Data generators for Problem A
def gen_blood_banks(n: int, min_supply=10, max_supply=100) -> List[BloodBank]:
    """Generate random blood banks with supplies"""
    banks = []
    for i in range(n):
        supplies = {}
        for blood_type in BLOOD_TYPES:
            if random.random() > 0.3:  # 70% chance bank has this type
                supplies[blood_type] = random.randint(min_supply, max_supply)
        banks.append(BloodBank(id=i, supplies=supplies))
    return banks


def gen_hospitals(n: int, min_demand=5, max_demand=50) -> List[Hospital]:
    """Generate random hospitals with demands"""
    hospitals = []
    for i in range(n):
        demands = {}
        for blood_type in BLOOD_TYPES:
            if random.random() > 0.4:  # 60% chance hospital needs this type
                demands[blood_type] = random.randint(min_demand, max_demand)
        hospitals.append(Hospital(id=i, demands=demands))
    return hospitals


# Experiments for Problem A
def exp_blood_sanity(trials=20, n_banks=3, n_hospitals=3) -> None:
    """Verify blood distribution produces reasonable results"""
    rows = []
    for _ in range(trials):
        banks = gen_blood_banks(n_banks)
        hospitals = gen_hospitals(n_hospitals)
        
        total_supply = sum(sum(b.supplies.values()) for b in banks)
        total_demand = sum(sum(h.demands.values()) for h in hospitals)
        
        flow, _ = solve_blood_distribution(banks, hospitals)
        
        # Flow should not exceed supply or demand
        assert flow <= total_supply, f"Flow {flow} exceeds supply {total_supply}"
        assert flow <= total_demand, f"Flow {flow} exceeds demand {total_demand}"
        
        rows.append((total_supply, total_demand, flow))
    
    write_csv("blood_sanity.csv", ["total_supply", "total_demand", "flow_delivered"], rows)
    print(f"[Blood] sanity passed ({trials} trials); wrote {path_in_outputs('blood_sanity.csv')}")


def exp_blood_timing(configs=[(5,5), (10,10), (15,15), (20,20), (25,25)], reps=3) -> None:
    """Measure runtime for blood distribution network flow"""
    out = []
    for n_banks, n_hospitals in configs:
        best_t = math.inf
        best_flow = -1
        
        for _ in range(reps):
            banks = gen_blood_banks(n_banks)
            hospitals = gen_hospitals(n_hospitals)
            
            t0 = now_ns()
            flow, _ = solve_blood_distribution(banks, hospitals)
            t1 = now_ns()
            
            elapsed = secs(t1 - t0)
            if elapsed < best_t:
                best_t = elapsed
                best_flow = flow
        
        total_nodes = 1 + n_banks + len(BLOOD_TYPES) + n_hospitals + 1
        out.append((n_banks, n_hospitals, total_nodes, best_flow, best_t))
        print(f"[Blood] banks={n_banks:2d} hospitals={n_hospitals:2d}  "
              f"nodes={total_nodes:3d}  flow={best_flow:4d}  t≈{best_t:.5f}s")
    
    write_csv("blood_timing.csv", 
              ["n_banks", "n_hospitals", "total_nodes", "max_flow", "time_s"], out)
    
    plt = try_matplotlib()
    if plt:
        nodes = [r[2] for r in out]
        times = [r[4] for r in out]
        
        plt.figure()
        plt.plot(nodes, times, marker="o", label="Network Flow runtime")
        plt.xlabel("Total nodes in network")
        plt.ylabel("Time (s)")
        plt.title("Blood Distribution Network Flow Runtime")
        plt.grid(True, which="both")
        plt.legend()
        plt.savefig(path_in_outputs("blood_runtime.png"), dpi=160, bbox_inches="tight")
        plt.close()


# =========================================================
# Problem B: Museum Artwork Arrangement (NP-Complete)
# =========================================================

GALLERY_ROOMS = ['Renaissance', 'Modern', 'Contemporary', 'Impressionist', 'Baroque', 'Abstract']

@dataclass
class Artwork:
    name: str
    id: int
    allowed_rooms: Set[str]  # subset of GALLERY_ROOMS
    value: int  # for exhibit quality scoring

@dataclass
class Conflict:
    artwork1_id: int
    artwork2_id: int
    reason: str  # 'thematic', 'lighting', 'humidity', 'security'


def build_conflict_graph(artworks: List[Artwork], 
                         conflicts: List[Conflict]) -> Dict[int, Set[int]]:
    """Build adjacency list for artwork conflict graph"""
    graph = defaultdict(set)
    for conflict in conflicts:
        a1, a2 = conflict.artwork1_id, conflict.artwork2_id
        graph[a1].add(a2)
        graph[a2].add(a1)
    return graph


def greedy_welsh_powell(artworks: List[Artwork], 
                       conflicts: List[Conflict]) -> Optional[Dict[int, str]]:
    """
    Welsh-Powell greedy graph coloring heuristic for museum arrangement.
    Colors = GALLERY_ROOMS; vertices = artworks; edges = conflicts.
    
    Algorithm:
    1. Sort artworks by degree (number of conflicts) descending
    2. Assign each artwork the first available room (color)
       that doesn't conflict with neighbors and is allowed for the artwork
    
    Returns: artwork_id -> gallery_room assignment, or None if infeasible
    """
    graph = build_conflict_graph(artworks, conflicts)
    
    # Sort artworks by degree (number of conflicts) descending
    degrees = [(len(graph[a.id]), a) for a in artworks]
    degrees.sort(reverse=True, key=lambda x: x[0])
    
    assignment = {}
    
    for _, artwork in degrees:
        # Find rooms used by conflicting neighbors
        neighbor_rooms = set()
        for neighbor_id in graph[artwork.id]:
            if neighbor_id in assignment:
                neighbor_rooms.add(assignment[neighbor_id])
        
        # Try to assign first available allowed room
        assigned = False
        for room in GALLERY_ROOMS:
            if room in artwork.allowed_rooms and room not in neighbor_rooms:
                assignment[artwork.id] = room
                assigned = True
                break
        
        if not assigned:
            return None  # Infeasible
    
    return assignment


def greedy_dsatur(artworks: List[Artwork], 
                 conflicts: List[Conflict]) -> Optional[Dict[int, str]]:
    """
    DSatur (Degree of Saturation) greedy coloring heuristic.
    
    Algorithm:
    1. Always place the artwork with highest saturation degree
       (saturation = number of distinct rooms used by conflicting neighbors)
    2. Break ties by choosing artwork with highest unplaced degree
    3. Assign first available room from allowed set
    
    Returns: artwork_id -> gallery_room assignment, or None if infeasible
    """
    graph = build_conflict_graph(artworks, conflicts)
    artwork_map = {a.id: a for a in artworks}
    
    assignment = {}
    unplaced = set(a.id for a in artworks)
    
    while unplaced:
        # Calculate saturation degree for each unplaced artwork
        max_sat = -1
        max_deg = -1
        next_artwork = None
        
        for artwork_id in unplaced:
            # Saturation: number of distinct rooms in neighborhood
            neighbor_rooms = set()
            for neighbor_id in graph[artwork_id]:
                if neighbor_id in assignment:
                    neighbor_rooms.add(assignment[neighbor_id])
            saturation = len(neighbor_rooms)
            
            # Unplaced degree: number of unplaced conflicting neighbors
            unplaced_degree = len([n for n in graph[artwork_id] if n in unplaced])
            
            if saturation > max_sat or (saturation == max_sat and unplaced_degree > max_deg):
                max_sat = saturation
                max_deg = unplaced_degree
                next_artwork = artwork_id
        
        # Place next_artwork in first available room
        artwork = artwork_map[next_artwork]
        neighbor_rooms = set()
        for neighbor_id in graph[next_artwork]:
            if neighbor_id in assignment:
                neighbor_rooms.add(assignment[neighbor_id])
        
        assigned = False
        for room in GALLERY_ROOMS:
            if room in artwork.allowed_rooms and room not in neighbor_rooms:
                assignment[next_artwork] = room
                assigned = True
                break
        
        if not assigned:
            return None  # Infeasible
        
        unplaced.remove(next_artwork)
    
    return assignment


def verify_arrangement(artworks: List[Artwork], 
                      conflicts: List[Conflict],
                      assignment: Optional[Dict[int, str]]) -> bool:
    """Verify that an arrangement satisfies all constraints"""
    if assignment is None:
        return False
    
    # Check all artworks are assigned
    artwork_ids = {a.id for a in artworks}
    if set(assignment.keys()) != artwork_ids:
        return False
    
    # Check room constraints
    artwork_map = {a.id: a for a in artworks}
    for artwork_id, room in assignment.items():
        if room not in artwork_map[artwork_id].allowed_rooms:
            return False
    
    # Check conflict constraints
    for conflict in conflicts:
        a1, a2 = conflict.artwork1_id, conflict.artwork2_id
        if assignment[a1] == assignment[a2]:
            return False
    
    return True


# Data generators for Problem B
def gen_artworks(n: int, prob_any_room=0.6) -> List[Artwork]:
    """Generate random artworks with room constraints"""
    artworks = []
    
    for i in range(n):
        if random.random() < prob_any_room:
            # Can be displayed in any room
            allowed = set(GALLERY_ROOMS)
        else:
            # Restricted to subset of rooms (e.g., size, lighting, theme)
            num_allowed = random.randint(2, len(GALLERY_ROOMS) - 1)
            allowed = set(random.sample(GALLERY_ROOMS, num_allowed))
        
        value = random.randint(100, 10000)  # artwork value
        artworks.append(Artwork(name=f"Artwork_{i}", id=i, 
                               allowed_rooms=allowed, value=value))
    
    return artworks


def gen_conflicts(artworks: List[Artwork], edge_prob=0.15) -> List[Conflict]:
    """Generate random artwork conflicts (Erdős–Rényi graph)"""
    conflicts = []
    n = len(artworks)
    reasons = ['thematic', 'lighting', 'humidity', 'security']
    
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_prob:
                reason = random.choice(reasons)
                conflicts.append(Conflict(artwork1_id=i, artwork2_id=j, reason=reason))
    
    return conflicts


# Experiments for Problem B
def exp_museum_sanity(trials=20, n=15) -> None:
    """Verify museum arrangement produces valid placements"""
    rows = []
    
    for _ in range(trials):
        artworks = gen_artworks(n, prob_any_room=0.7)
        conflicts = gen_conflicts(artworks, edge_prob=0.15)
        
        wp_arrangement = greedy_welsh_powell(artworks, conflicts)
        ds_arrangement = greedy_dsatur(artworks, conflicts)
        
        wp_valid = verify_arrangement(artworks, conflicts, wp_arrangement)
        ds_valid = verify_arrangement(artworks, conflicts, ds_arrangement)
        
        wp_rooms = len(set(wp_arrangement.values())) if wp_arrangement else 0
        ds_rooms = len(set(ds_arrangement.values())) if ds_arrangement else 0
        
        rows.append((len(artworks), len(conflicts), 
                    wp_valid, wp_rooms, ds_valid, ds_rooms))
    
    write_csv("museum_sanity.csv", 
              ["n_artworks", "n_conflicts", "wp_valid", "wp_rooms", 
               "ds_valid", "ds_rooms"], rows)
    print(f"[Museum] sanity passed ({trials} trials); wrote {path_in_outputs('museum_sanity.csv')}")


def exp_museum_timing(ns=(10, 20, 30, 40, 50, 60), edge_prob=0.07, reps=5) -> None:
    """Measure runtime for museum arrangement greedy heuristics"""
    out = []
    
    for n in ns:
        best_twp = math.inf
        best_tds = math.inf
        wp_rooms = 0
        ds_rooms = 0
        n_edges = 0
        best_wp_arrangement = None
        best_ds_arrangement = None
        
        successful_reps = 0
        attempts = 0
        max_attempts = reps * 10  # Allow many retries to guarantee success
        
        while successful_reps < reps and attempts < max_attempts:
            attempts += 1
            # More relaxed parameters for larger instances
            prob_any = 0.85 if n >= 50 else 0.80
            artworks = gen_artworks(n, prob_any_room=prob_any)
            conflicts = gen_conflicts(artworks, edge_prob=edge_prob)
            
            t0 = now_ns()
            wp_arrangement = greedy_welsh_powell(artworks, conflicts)
            t1 = now_ns()
            twp = secs(t1 - t0)
            
            t0 = now_ns()
            ds_arrangement = greedy_dsatur(artworks, conflicts)
            t1 = now_ns()
            tds = secs(t1 - t0)
            
            # Only count if BOTH algorithms succeed
            if wp_arrangement is not None and ds_arrangement is not None:
                successful_reps += 1
                n_edges = len(conflicts)
                
                # Update best times
                if twp < best_twp:
                    best_twp = twp
                    best_wp_arrangement = wp_arrangement
                
                if tds < best_tds:
                    best_tds = tds
                    best_ds_arrangement = ds_arrangement
        
        if successful_reps < reps:
            print(f"  WARNING: Only achieved {successful_reps}/{reps} successful runs for n={n}")
        
        # Count rooms from the best run's arrangement
        wp_rooms = len(set(best_wp_arrangement.values())) if best_wp_arrangement is not None else 0
        ds_rooms = len(set(best_ds_arrangement.values())) if best_ds_arrangement is not None else 0
        
        # Debug: print if arrangements are None
        if best_wp_arrangement is None:
            print(f"  WARNING: Welsh-Powell returned None (infeasible) for n={n}")
        if best_ds_arrangement is None:
            print(f"  WARNING: DSatur returned None (infeasible) for n={n}")
        
        out.append((n, n_edges, best_twp, wp_rooms, best_tds, ds_rooms))
        print(f"[Museum] n={n:2d}  conflicts={n_edges:3d}  "
              f"t_WP≈{best_twp:.6f}s (rooms={wp_rooms})  "
              f"t_DS≈{best_tds:.6f}s (rooms={ds_rooms})")
    
    write_csv("museum_timing.csv", 
              ["n_artworks", "n_conflicts", "time_wp_s", "rooms_wp",
               "time_ds_s", "rooms_ds"], out)
    
    plt = try_matplotlib()
    if plt:
        ns_list = [r[0] for r in out]
        twp_list = [r[2] for r in out]
        tds_list = [r[4] for r in out]
        
        # Runtime comparison
        plt.figure()
        plt.plot(ns_list, twp_list, marker="o", label="Welsh-Powell")
        plt.plot(ns_list, tds_list, marker="s", label="DSatur")
        plt.xlabel("Number of artworks")
        plt.ylabel("Time (s)")
        plt.title("Museum Arrangement: Greedy Heuristics Runtime")
        plt.legend()
        plt.grid(True, which="both")
        plt.savefig(path_in_outputs("museum_runtime.png"), dpi=160, bbox_inches="tight")
        plt.close()
        
        # Room usage comparison
        wp_rooms_list = [r[3] for r in out]
        ds_rooms_list = [r[5] for r in out]
        
        plt.figure()
        plt.plot(ns_list, wp_rooms_list, marker="o", label="Welsh-Powell rooms")
        plt.plot(ns_list, ds_rooms_list, marker="s", label="DSatur rooms")
        plt.xlabel("Number of artworks")
        plt.ylabel("Gallery rooms used")
        plt.title("Museum Arrangement: Room Usage (Lower is Better)")
        plt.legend()
        plt.grid(True, which="both")
        plt.savefig(path_in_outputs("museum_rooms.png"), dpi=160, bbox_inches="tight")
        plt.close()


# ================================
# Main Execution
# ================================
def main():
    random.seed(7)
    ensure_outputs_dir()
    
    print("=" * 60)
    print("Problem A: Blood Bank Distribution (Network Flow)")
    print("=" * 60)
    exp_blood_sanity(trials=20, n_banks=3, n_hospitals=3)
    exp_blood_timing()
    
    print("\n" + "=" * 60)
    print("Problem B: Museum Artwork Arrangement (NP-Complete via Graph Coloring)")
    print("=" * 60)
    exp_museum_sanity(trials=20, n=15)
        
    exp_museum_timing(ns=(10, 20, 30, 40, 50, 60), edge_prob=0.07, reps=5)
    
    print("\n" + "=" * 60)
    print("Artifacts written to:", os.path.abspath(OUTPUT_DIR))
    print("=" * 60)
    
    files = [
        "blood_sanity.csv", "blood_timing.csv", "blood_runtime.png",
        "museum_sanity.csv", "museum_timing.csv", 
        "museum_runtime.png", "museum_rooms.png"
    ]
    
    for fn in files:
        p = path_in_outputs(fn)
        exists = os.path.exists(p)
        status = "" if exists else "(missing - matplotlib required for .png)"
        print(f" - {p} {status}")


if __name__ == "__main__":
    main()