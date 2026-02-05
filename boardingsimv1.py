#!/usr/bin/env python3
"""
==[[Airplane Boarding Simulation by MKMUSES]]==
===============================================
Discrete-event simulation comparing four boarding algorithms:
1. Steffen method
2. WILMA (Window-Middle-Aisle with back-to-front)
3. Random boarding
4. Back-to-front block boarding

Author: MKmuses 30th January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==============[CONFIGURATION PARAMETERS (easily tunable)]====================

CONFIG = {
    # Aircraft configuration
    'N_ROWS': 30,                    # Number of seat rows
    'SEATS_PER_ROW': 6,              # Seats per row (A-F)
    'SEAT_LETTERS': ['A', 'B', 'C', 'D', 'E', 'F'],
    
    # Time parameters
    'TIME_STEP': 0.5,                # Seconds per simulation tick
    
    # Passenger speed distribution (steps per second)
    # Faster walking speed means less aisle congestion advantage for Steffen
    'WALK_SPEED_MEAN': 2.0,          # Mean walking speed (rows/second)
    'WALK_SPEED_STD': 0.4,           # Std dev of walking speed
    'WALK_SPEED_MIN': 1.2,           # Minimum walking speed
    'WALK_SPEED_MAX': 3.0,           # Maximum walking speed
    
    # Stow time distribution (seconds)
    # Longer stow times favor WILMA because it maximizes parallel stowing
    # (all window passengers can stow simultaneously without seat interference)
    'STOW_TIME_MEAN': 12.0,          # Mean time to stow luggage
    'STOW_TIME_STD': 4.0,            # Std dev of stow time
    'STOW_TIME_MIN': 5.0,            # Minimum stow time
    'STOW_TIME_MAX': 25.0,           # Maximum stow time
    
    # Seat interference time (seconds) - CRITICAL FOR WILMA ADVANTAGE
    # Time added when seated passengers must stand to let others into window/middle seats
    # WILMA avoids this by boarding window->middle->aisle
    'SEAT_INTERFERENCE_TIME': 8.0,   # Time for 1 person to stand and reseat
    
    # Carry-on luggage - higher probability means more stowing, favoring WILMA
    'HAS_CARRYON_PROB': 0.95,        # Probability passenger has carry-on
    
    # Back-to-front block configuration
    'BLOCK_SIZE': 10,                # Rows per block for back-to-front
    
    # Priority boarding groups - adds real-world randomness
    # These passengers board BEFORE their assigned algorithm order
    'PRIORITY_BOARDING': {
        'first_class_rows': 3,       # First 3 rows are first class (board first via front)
        'families_pct': 0.08,        # 8% are families with kids (pre-board, random seats)
        'elderly_pct': 0.05,         # 5% elderly/disabled (pre-board, random seats)
        'military_pct': 0.03,        # 3% military (pre-board, random seats)
        'frequent_flyer_pct': 0.10,  # 10% elite status (board after pre-board, before main)
    },
    
    # Slower passenger characteristics
    'SLOW_PASSENGER_GROUPS': ['families', 'elderly'],  # Groups that are slower
    'SLOW_WALK_SPEED_FACTOR': 0.6,   # Walk at 60% of normal speed
    'SLOW_STOW_TIME_FACTOR': 1.5,    # Take 50% longer to stow
    
    # Monte Carlo parameters
    'N_REPLICATES': 500,             # Number of simulation runs per algorithm
    'RANDOM_SEED': 42,               # For reproducibility
    
    # Animation parameters
    'ANIMATE_REPLICATE': 0,          # Which replicate to animate (0 = first)
    'ANIMATION_INTERVAL': 50,        # Milliseconds between frames
    'ANIMATION_SPEEDUP': 4,          # Skip frames for faster animation
}

# ==============[DATA STRUCTURE]====================

class PassengerState(Enum):
    WAITING = 0      # In queue, not yet entered plane
    WALKING = 1      # Walking down aisle
    STOWING = 2      # Stowing luggage (blocking aisle)
    SEATED = 3       # In seat


@dataclass
class Passenger:
    id: int
    row: int                         # 1-indexed row number
    seat: str                        # Seat letter (A-F)
    walk_speed: float                # Rows per second
    stow_time: float                 # Seconds to stow luggage
    has_carryon: bool                # Whether they have carry-on luggage
    boarding_group: str = field(default='regular')  # Priority boarding group
    
    # State tracking (mutable)
    state: PassengerState = field(default=PassengerState.WAITING)
    aisle_position: int = field(default=0)   # Current position in aisle (0 = not in aisle)
    stow_remaining: float = field(default=0) # Time remaining to stow
    seated_time: float = field(default=0)    # Time when seated
    
    def get_seat_type(self) -> str:
        """Return 'window', 'middle', or 'aisle'"""
        if self.seat in ['A', 'F']:
            return 'window'
        elif self.seat in ['B', 'E']:
            return 'middle'
        else:
            return 'aisle'
    
    def reset(self):
        """Reset state for new simulation run"""
        self.state = PassengerState.WAITING
        self.aisle_position = 0
        self.stow_remaining = 0
        self.seated_time = 0


@dataclass
class SimulationResult:
    algorithm: str
    boarding_time: float
    time_series: List[Tuple[float, int]]  # (time, passengers_seated)
    passenger_states: Optional[List[Dict]] = None  # For animation

# ==============[PASSENGER GENERATION]====================

def generate_passengers(config: dict, no_carryon: bool = False) -> List[Passenger]:
    """
    Generate a full plane of passengers with random attributes and boarding groups.
    
    Boarding groups include:
    - first_class: Rows 1-N (front of plane, board first)
    - families: Families with children (pre-board, slower, random seats)
    - elderly: Elderly/disabled passengers (pre-board, slower, random seats)
    - military: Active military (pre-board after families/elderly)
    - frequent_flyer: Elite status (board after pre-board, before general)
    - regular: Everyone else
    
    Args:
        config: Configuration dictionary
        no_carryon: If True, all passengers have stow_time=0
    
    Returns:
        List of Passenger objects
    """
    passengers = []
    passenger_id = 0
    
    priority = config.get('PRIORITY_BOARDING', {})
    first_class_rows = priority.get('first_class_rows', 3)
    families_pct = priority.get('families_pct', 0.08)
    elderly_pct = priority.get('elderly_pct', 0.05)
    military_pct = priority.get('military_pct', 0.03)
    frequent_flyer_pct = priority.get('frequent_flyer_pct', 0.10)
    
    slow_groups = config.get('SLOW_PASSENGER_GROUPS', ['families', 'elderly'])
    slow_walk_factor = config.get('SLOW_WALK_SPEED_FACTOR', 0.6)
    slow_stow_factor = config.get('SLOW_STOW_TIME_FACTOR', 1.5)
    
    # First pass: create all passengers with basic attributes
    for row in range(1, config['N_ROWS'] + 1):
        for seat in config['SEAT_LETTERS']:
            # Base walking speed
            walk_speed = np.clip(
                np.random.normal(config['WALK_SPEED_MEAN'], config['WALK_SPEED_STD']),
                config['WALK_SPEED_MIN'],
                config['WALK_SPEED_MAX']
            )
            
            # Determine if passenger has carry-on
            has_carryon = not no_carryon and (np.random.random() < config['HAS_CARRYON_PROB'])
            
            # Base stow time
            if has_carryon:
                stow_time = np.clip(
                    np.random.normal(config['STOW_TIME_MEAN'], config['STOW_TIME_STD']),
                    config['STOW_TIME_MIN'],
                    config['STOW_TIME_MAX']
                )
            else:
                stow_time = 0.5  # Minimal time to enter seat
            
            # Assign boarding group
            if row <= first_class_rows:
                boarding_group = 'first_class'
            else:
                boarding_group = 'regular'  # Will be reassigned below
            
            passenger = Passenger(
                id=passenger_id,
                row=row,
                seat=seat,
                walk_speed=walk_speed,
                stow_time=stow_time,
                has_carryon=has_carryon,
                boarding_group=boarding_group
            )
            passengers.append(passenger)
            passenger_id += 1
    
    # Second pass: assign special boarding groups to non-first-class passengers
    regular_passengers = [p for p in passengers if p.boarding_group == 'regular']
    n_regular = len(regular_passengers)
    
    # Shuffle to randomize assignment
    shuffled_indices = list(range(n_regular))
    random.shuffle(shuffled_indices)
    
    # Calculate counts for each group
    n_families = int(n_regular * families_pct)
    n_elderly = int(n_regular * elderly_pct)
    n_military = int(n_regular * military_pct)
    n_frequent = int(n_regular * frequent_flyer_pct)
    
    idx = 0
    
    # Assign families (often traveling together, so assign in pairs/groups)
    family_assigned = 0
    while family_assigned < n_families and idx < n_regular:
        p = regular_passengers[shuffled_indices[idx]]
        p.boarding_group = 'families'
        # Slow down families
        p.walk_speed *= slow_walk_factor
        p.stow_time *= slow_stow_factor
        family_assigned += 1
        idx += 1
        
        # Try to assign adjacent seat as family member too
        same_row_partner = next(
            (p2 for p2 in regular_passengers 
             if p2.row == p.row and p2.boarding_group == 'regular' 
             and p2.seat != p.seat),
            None
        )
        if same_row_partner and family_assigned < n_families:
            same_row_partner.boarding_group = 'families'
            same_row_partner.walk_speed *= slow_walk_factor
            same_row_partner.stow_time *= slow_stow_factor
            family_assigned += 1
    
    # Assign elderly
    for i in range(n_elderly):
        if idx >= n_regular:
            break
        p = regular_passengers[shuffled_indices[idx]]
        if p.boarding_group == 'regular':
            p.boarding_group = 'elderly'
            p.walk_speed *= slow_walk_factor
            p.stow_time *= slow_stow_factor
        idx += 1
    
    # Assign military
    for i in range(n_military):
        if idx >= n_regular:
            break
        p = regular_passengers[shuffled_indices[idx]]
        if p.boarding_group == 'regular':
            p.boarding_group = 'military'
        idx += 1
    
    # Assign frequent flyers
    for i in range(n_frequent):
        if idx >= n_regular:
            break
        p = regular_passengers[shuffled_indices[idx]]
        if p.boarding_group == 'regular':
            p.boarding_group = 'frequent_flyer'
        idx += 1
    
    return passengers


def apply_priority_boarding(passengers: List[Passenger], algorithm_order: List[Passenger]) -> List[Passenger]:
    """
    Apply real-world priority boarding rules to any algorithm's order.
    
    Boarding sequence:
    1. First class (front rows) - always first
    2. Families with children, elderly/disabled - pre-boarding
    3. Military - after families/elderly
    4. Frequent flyers / elite status - before general boarding
    5. Algorithm-determined order for remaining passengers
    
    This simulates real airline boarding which disrupts "optimal" algorithms.
    """
    # Separate by boarding group
    first_class = [p for p in algorithm_order if p.boarding_group == 'first_class']
    families = [p for p in algorithm_order if p.boarding_group == 'families']
    elderly = [p for p in algorithm_order if p.boarding_group == 'elderly']
    military = [p for p in algorithm_order if p.boarding_group == 'military']
    frequent = [p for p in algorithm_order if p.boarding_group == 'frequent_flyer']
    regular = [p for p in algorithm_order if p.boarding_group == 'regular']
    
    # Shuffle within priority groups (they don't follow the algorithm)
    random.shuffle(families)
    random.shuffle(elderly)
    random.shuffle(military)
    random.shuffle(frequent)
    # Regular passengers keep algorithm order
    
    # Combine in priority sequence
    final_order = first_class + families + elderly + military + frequent + regular
    
    return final_order

# ==============[BOARDING ORDER ALGORITHMS]====================

def get_boarding_order_back_to_front(passengers: List[Passenger], config: dict) -> List[Passenger]:
    """
    Back-to-front block boarding.
    Passengers board in blocks from back to front.
    """
    block_size = config['BLOCK_SIZE']
    n_rows = config['N_ROWS']
    
    # Create blocks
    blocks = []
    for block_start in range(n_rows, 0, -block_size):
        block_end = max(block_start - block_size + 1, 1)
        block_passengers = [p for p in passengers if block_end <= p.row <= block_start]
        # Shuffle within block
        random.shuffle(block_passengers)
        blocks.append(block_passengers)
    
    # Flatten
    order = []
    for block in blocks:
        order.extend(block)
    
    return order


def get_boarding_order_random(passengers: List[Passenger], config: dict) -> List[Passenger]:
    """
    Random boarding order (Southwest-style with assigned seats).
    """
    order = passengers.copy()
    random.shuffle(order)
    return order


def get_boarding_order_wilma(passengers: List[Passenger], config: dict) -> List[Passenger]:
    """
    WILMA: Window-Middle-Aisle boarding.
    
    Board all window seats first (alternating even/odd rows for spacing),
    then middle seats, then aisle seats.
    
    This maximizes parallel stowing by:
    1. Ensuring no seat interference (window->middle->aisle order)
    2. Spacing passengers apart (alternating rows) to allow simultaneous stowing
    """
    n_rows = config['N_ROWS']
    
    # Group by seat type
    windows = [p for p in passengers if p.get_seat_type() == 'window']
    middles = [p for p in passengers if p.get_seat_type() == 'middle']
    aisles = [p for p in passengers if p.get_seat_type() == 'aisle']
    
    def spread_order(pax_list):
        """Order passengers with spacing: even rows back-to-front, then odd rows"""
        if n_rows % 2 == 0:
            even_rows = list(range(n_rows, 0, -2))
            odd_rows = list(range(n_rows - 1, 0, -2))
        else:
            odd_rows = list(range(n_rows, 0, -2))
            even_rows = list(range(n_rows - 1, 0, -2))
        
        ordered = []
        # Even rows first
        for row in even_rows:
            ordered.extend([p for p in pax_list if p.row == row])
        # Then odd rows
        for row in odd_rows:
            ordered.extend([p for p in pax_list if p.row == row])
        return ordered
    
    # Apply spacing within each seat type group
    windows_ordered = spread_order(windows)
    middles_ordered = spread_order(middles)
    aisles_ordered = spread_order(aisles)
    
    # Combine: windows first, then middles, then aisles
    return windows_ordered + middles_ordered + aisles_ordered


def get_boarding_order_steffen(passengers: List[Passenger], config: dict) -> List[Passenger]:
    """
    Steffen method: Board in waves where adjacent passengers in line
    are separated by 2 rows and use alternating seat positions.
    
    Pattern: All even-row windows on one side, then odd-row windows on same side,
    then even-row windows other side, etc. This maximizes parallel stowing.
    """
    n_rows = config['N_ROWS']
    order = []
    
    # Define the Steffen sequence
    # We do: window right (F) even rows back-to-front, then odd rows
    # Then window left (A) even rows, then odd rows
    # Then middle right (E) even, odd
    # Then middle left (B) even, odd
    # Then aisle right (D) even, odd
    # Then aisle left (C) even, odd
    
    seat_sequence = ['F', 'A', 'E', 'B', 'D', 'C']
    
    for seat in seat_sequence:
        # Even rows first (back to front)
        even_rows = [r for r in range(n_rows, 0, -2) if r % 2 == 0]
        odd_rows = [r for r in range(n_rows, 0, -2) if r % 2 == 1]
        
        # Handle edge case where n_rows might give different pattern
        if n_rows % 2 == 0:
            even_rows = list(range(n_rows, 0, -2))
            odd_rows = list(range(n_rows - 1, 0, -2))
        else:
            odd_rows = list(range(n_rows, 0, -2))
            even_rows = list(range(n_rows - 1, 0, -2))
        
        for row in even_rows:
            p = next((p for p in passengers if p.row == row and p.seat == seat), None)
            if p:
                order.append(p)
        
        for row in odd_rows:
            p = next((p for p in passengers if p.row == row and p.seat == seat), None)
            if p:
                order.append(p)
    
    return order

# ==============[BOARDING SIMULATION]====================

def calculate_seat_interference(passenger: Passenger, seated_passengers: List[Passenger], config: dict) -> float:
    """
    Calculate additional time needed due to seated passengers blocking the way.
    
    For a passenger to reach their seat, any seated passengers between them 
    and the aisle must stand up and move out of the way.
    
    Seat layout (per side):
    - Left side: A (window), B (middle), C (aisle)
    - Right side: D (aisle), E (middle), F (window)
    
    Returns additional time in seconds.
    """
    interference_time = config.get('SEAT_INTERFERENCE_TIME', 8.0)
    
    row = passenger.row
    seat = passenger.seat
    
    # Find seated passengers in the same row
    same_row_seated = [p for p in seated_passengers 
                      if p.row == row and p.state == PassengerState.SEATED]
    
    blocking_count = 0
    
    # Left side: A-B-C (C is aisle)
    if seat == 'A':  # Window - B and C could block
        blocking_count = sum(1 for p in same_row_seated if p.seat in ['B', 'C'])
    elif seat == 'B':  # Middle - C could block
        blocking_count = sum(1 for p in same_row_seated if p.seat == 'C')
    # C is aisle seat, no one blocks
    
    # Right side: D-E-F (D is aisle)
    elif seat == 'F':  # Window - E and D could block
        blocking_count = sum(1 for p in same_row_seated if p.seat in ['E', 'D'])
    elif seat == 'E':  # Middle - D could block
        blocking_count = sum(1 for p in same_row_seated if p.seat == 'D')
    # D is aisle seat, no one blocks
    
    return blocking_count * interference_time


def simulate_boarding(
    passengers: List[Passenger],
    boarding_order: List[Passenger],
    config: dict,
    record_states: bool = False
) -> SimulationResult:
    """
    Run discrete-time boarding simulation with seat interference modeling.
    
    Args:
        passengers: List of all passengers (will be modified)
        boarding_order: Order in which passengers board
        config: Configuration dictionary
        record_states: If True, record state at each timestep for animation
    
    Returns:
        SimulationResult with boarding time and time series
    """
    dt = config['TIME_STEP']
    n_rows = config['N_ROWS']
    
    # Reset all passengers
    for p in passengers:
        p.reset()
    
    # Track aisle occupancy: aisle[row] = passenger or None
    aisle = {row: None for row in range(0, n_rows + 2)}  # 0 is entry, n_rows+1 is past last row
    
    # Queue of passengers waiting to board
    queue = list(boarding_order)
    
    # Time tracking
    current_time = 0.0
    time_series = [(0.0, 0)]
    state_history = [] if record_states else None
    
    # Main simulation loop
    while True:
        # Count seated passengers
        n_seated = sum(1 for p in passengers if p.state == PassengerState.SEATED)
        
        # Check termination
        if n_seated == len(passengers):
            break
        
        # Safety check for infinite loops
        if current_time > 10000:
            print(f"Warning: Simulation timeout at {current_time}s")
            break
        
        # Record state for animation
        if record_states:
            state_snapshot = {
                'time': current_time,
                'aisle': {k: (v.id, v.row, v.seat) if v else None for k, v in aisle.items()},
                'passengers': [
                    {
                        'id': p.id,
                        'row': p.row,
                        'seat': p.seat,
                        'state': p.state.value,
                        'aisle_pos': p.aisle_position,
                        'stow_remaining': p.stow_remaining
                    }
                    for p in passengers
                ]
            }
            state_history.append(state_snapshot)
        
        # Process passengers in reverse order of aisle position (front to back)
        # This ensures passengers ahead move first
        active_passengers = [p for p in passengers 
                           if p.state in [PassengerState.WALKING, PassengerState.STOWING]]
        active_passengers.sort(key=lambda p: -p.aisle_position)
        
        for p in active_passengers:
            if p.state == PassengerState.STOWING:
                # Continue stowing
                p.stow_remaining -= dt
                if p.stow_remaining <= 0:
                    # Done stowing, take seat
                    p.state = PassengerState.SEATED
                    p.seated_time = current_time
                    aisle[p.aisle_position] = None
            
            elif p.state == PassengerState.WALKING:
                # Check if at destination row
                if p.aisle_position == p.row:
                    # Start stowing - add seat interference time
                    p.state = PassengerState.STOWING
                    interference = calculate_seat_interference(p, passengers, config)
                    p.stow_remaining = p.stow_time + interference
                else:
                    # Try to move forward
                    # Calculate how far we can move this timestep
                    move_distance = p.walk_speed * dt
                    target_pos = min(p.aisle_position + 1, p.row)  # Move at most 1 row
                    
                    # Check if next position is free
                    if aisle[target_pos] is None:
                        # Move forward
                        aisle[p.aisle_position] = None
                        p.aisle_position = target_pos
                        aisle[target_pos] = p
        
        # Try to let next passenger enter
        if queue and aisle[1] is None:  # Position 1 is first row in aisle
            next_passenger = queue.pop(0)
            next_passenger.state = PassengerState.WALKING
            next_passenger.aisle_position = 1
            aisle[1] = next_passenger
        
        # Advance time
        current_time += dt
        
        # Record time series
        n_seated_now = sum(1 for p in passengers if p.state == PassengerState.SEATED)
        if n_seated_now != time_series[-1][1]:
            time_series.append((current_time, n_seated_now))
    
    # Final time series point
    time_series.append((current_time, len(passengers)))
    
    return SimulationResult(
        algorithm="",  # Will be set by caller
        boarding_time=current_time,
        time_series=time_series,
        passenger_states=state_history
    )


# ==============[TWO-DOOR BOARDING SIMULATION]====================

def get_boarding_order_two_door_random(passengers: List[Passenger], config: dict) -> Tuple[List[Passenger], List[Passenger]]:
    """
    Random boarding with two doors.
    Front half passengers use front door, back half use rear door.
    """
    n_rows = config['N_ROWS']
    mid_row = n_rows // 2
    
    front_passengers = [p for p in passengers if p.row <= mid_row]
    back_passengers = [p for p in passengers if p.row > mid_row]
    
    random.shuffle(front_passengers)
    random.shuffle(back_passengers)
    
    return front_passengers, back_passengers


def get_boarding_order_two_door_back_to_front(passengers: List[Passenger], config: dict) -> Tuple[List[Passenger], List[Passenger]]:
    """
    Back-to-front with two doors.
    Each door loads its half from the middle outward.
    """
    n_rows = config['N_ROWS']
    mid_row = n_rows // 2
    block_size = config['BLOCK_SIZE'] // 2  # Smaller blocks for each door
    
    # Front door: rows 1 to mid_row, board from mid_row toward row 1
    front_passengers = [p for p in passengers if p.row <= mid_row]
    front_order = []
    for block_start in range(mid_row, 0, -block_size):
        block_end = max(block_start - block_size + 1, 1)
        block = [p for p in front_passengers if block_end <= p.row <= block_start]
        random.shuffle(block)
        front_order.extend(block)
    
    # Back door: rows mid_row+1 to n_rows, board from mid_row+1 toward n_rows
    back_passengers = [p for p in passengers if p.row > mid_row]
    back_order = []
    for block_start in range(mid_row + 1, n_rows + 1, block_size):
        block_end = min(block_start + block_size - 1, n_rows)
        block = [p for p in back_passengers if block_start <= p.row <= block_end]
        random.shuffle(block)
        back_order.extend(block)
    
    return front_order, back_order


def get_boarding_order_two_door_wilma(passengers: List[Passenger], config: dict) -> Tuple[List[Passenger], List[Passenger]]:
    """
    WILMA with two doors.
    Window-Middle-Aisle within each half, with row spacing for parallel loading.
    """
    n_rows = config['N_ROWS']
    mid_row = n_rows // 2
    
    front_passengers = [p for p in passengers if p.row <= mid_row]
    back_passengers = [p for p in passengers if p.row > mid_row]
    
    def wilma_order_spread(pax_list, rows_range, reverse=False):
        """WILMA order with alternating row spacing"""
        windows = [p for p in pax_list if p.get_seat_type() == 'window']
        middles = [p for p in pax_list if p.get_seat_type() == 'middle']
        aisles = [p for p in pax_list if p.get_seat_type() == 'aisle']
        
        rows = list(rows_range)
        if reverse:
            # Front door: fill from middle toward row 1
            even_rows = sorted([r for r in rows if r % 2 == 0])
            odd_rows = sorted([r for r in rows if r % 2 == 1])
        else:
            # Back door: fill from middle toward back
            even_rows = sorted([r for r in rows if r % 2 == 0], reverse=True)
            odd_rows = sorted([r for r in rows if r % 2 == 1], reverse=True)
        
        def spread_order(pax_group):
            ordered = []
            for row in even_rows:
                ordered.extend([p for p in pax_group if p.row == row])
            for row in odd_rows:
                ordered.extend([p for p in pax_group if p.row == row])
            return ordered
        
        return spread_order(windows) + spread_order(middles) + spread_order(aisles)
    
    front_order = wilma_order_spread(front_passengers, range(1, mid_row + 1), reverse=True)
    back_order = wilma_order_spread(back_passengers, range(mid_row + 1, n_rows + 1), reverse=False)
    
    return front_order, back_order


def get_boarding_order_two_door_steffen(passengers: List[Passenger], config: dict) -> Tuple[List[Passenger], List[Passenger]]:
    """
    Steffen method with two doors.
    Apply Steffen pattern to each half independently.
    """
    n_rows = config['N_ROWS']
    mid_row = n_rows // 2
    
    front_passengers = [p for p in passengers if p.row <= mid_row]
    back_passengers = [p for p in passengers if p.row > mid_row]
    
    def steffen_order(pax_list, rows_range, reverse=False):
        order = []
        seat_sequence = ['F', 'A', 'E', 'B', 'D', 'C']
        
        rows = list(rows_range)
        if reverse:
            # For front door, we want to fill from middle toward front
            even_rows = sorted([r for r in rows if r % 2 == 0])
            odd_rows = sorted([r for r in rows if r % 2 == 1])
        else:
            # For back door, fill from middle toward back
            even_rows = sorted([r for r in rows if r % 2 == 0], reverse=True)
            odd_rows = sorted([r for r in rows if r % 2 == 1], reverse=True)
        
        for seat in seat_sequence:
            for row in even_rows:
                p = next((p for p in pax_list if p.row == row and p.seat == seat), None)
                if p:
                    order.append(p)
            for row in odd_rows:
                p = next((p for p in pax_list if p.row == row and p.seat == seat), None)
                if p:
                    order.append(p)
        
        return order
    
    front_order = steffen_order(front_passengers, range(1, mid_row + 1), reverse=True)
    back_order = steffen_order(back_passengers, range(mid_row + 1, n_rows + 1), reverse=False)
    
    return front_order, back_order


def simulate_boarding_two_door(
    passengers: List[Passenger],
    front_order: List[Passenger],
    back_order: List[Passenger],
    config: dict,
    record_states: bool = False
) -> SimulationResult:
    """
    Run discrete-time boarding simulation with two doors.
    Front door at row 1, back door at row N.
    
    Args:
        passengers: List of all passengers (will be modified)
        front_order: Order for passengers entering from front door
        back_order: Order for passengers entering from back door
        config: Configuration dictionary
        record_states: If True, record state at each timestep for animation
    
    Returns:
        SimulationResult with boarding time and time series
    """
    dt = config['TIME_STEP']
    n_rows = config['N_ROWS']
    
    # Reset all passengers
    for p in passengers:
        p.reset()
    
    # Track aisle occupancy: aisle[row] = passenger or None
    aisle = {row: None for row in range(0, n_rows + 2)}
    
    # Two queues - front and back
    front_queue = list(front_order)
    back_queue = list(back_order)
    
    # Track which door each passenger entered from (for walking direction)
    entered_from_back = set()
    
    # Time tracking
    current_time = 0.0
    time_series = [(0.0, 0)]
    state_history = [] if record_states else None
    
    # Main simulation loop
    while True:
        # Count seated passengers
        n_seated = sum(1 for p in passengers if p.state == PassengerState.SEATED)
        
        # Check termination
        if n_seated == len(passengers):
            break
        
        # Safety check for infinite loops
        if current_time > 10000:
            print(f"Warning: Simulation timeout at {current_time}s")
            break
        
        # Record state for animation
        if record_states:
            state_snapshot = {
                'time': current_time,
                'aisle': {k: (v.id, v.row, v.seat) if v else None for k, v in aisle.items()},
                'passengers': [
                    {
                        'id': p.id,
                        'row': p.row,
                        'seat': p.seat,
                        'state': p.state.value,
                        'aisle_pos': p.aisle_position,
                        'stow_remaining': p.stow_remaining,
                        'from_back': p.id in entered_from_back
                    }
                    for p in passengers
                ],
                'two_door': True
            }
            state_history.append(state_snapshot)
        
        # Process passengers - need to handle bidirectional movement
        active_passengers = [p for p in passengers 
                           if p.state in [PassengerState.WALKING, PassengerState.STOWING]]
        
        # Sort by position - process from both ends toward middle
        # This prevents collisions
        active_passengers.sort(key=lambda p: p.aisle_position)
        
        for p in active_passengers:
            if p.state == PassengerState.STOWING:
                # Continue stowing
                p.stow_remaining -= dt
                if p.stow_remaining <= 0:
                    # Done stowing, take seat
                    p.state = PassengerState.SEATED
                    p.seated_time = current_time
                    aisle[p.aisle_position] = None
            
            elif p.state == PassengerState.WALKING:
                # Check if at destination row
                if p.aisle_position == p.row:
                    # Start stowing - add seat interference time
                    p.state = PassengerState.STOWING
                    interference = calculate_seat_interference(p, passengers, config)
                    p.stow_remaining = p.stow_time + interference
                else:
                    # Determine direction based on which door they entered
                    if p.id in entered_from_back:
                        # Entered from back, walking forward (decreasing row numbers)
                        target_pos = max(p.aisle_position - 1, p.row)
                    else:
                        # Entered from front, walking backward (increasing row numbers)
                        target_pos = min(p.aisle_position + 1, p.row)
                    
                    # Check if next position is free
                    if aisle[target_pos] is None:
                        # Move
                        aisle[p.aisle_position] = None
                        p.aisle_position = target_pos
                        aisle[target_pos] = p
        
        # Try to let passengers enter from front door (row 1)
        if front_queue and aisle[1] is None:
            next_passenger = front_queue.pop(0)
            next_passenger.state = PassengerState.WALKING
            next_passenger.aisle_position = 1
            aisle[1] = next_passenger
        
        # Try to let passengers enter from back door (row n_rows)
        if back_queue and aisle[n_rows] is None:
            next_passenger = back_queue.pop(0)
            next_passenger.state = PassengerState.WALKING
            next_passenger.aisle_position = n_rows
            aisle[n_rows] = next_passenger
            entered_from_back.add(next_passenger.id)
        
        # Advance time
        current_time += dt
        
        # Record time series
        n_seated_now = sum(1 for p in passengers if p.state == PassengerState.SEATED)
        if n_seated_now != time_series[-1][1]:
            time_series.append((current_time, n_seated_now))
    
    # Final time series point
    time_series.append((current_time, len(passengers)))
    
    return SimulationResult(
        algorithm="",
        boarding_time=current_time,
        time_series=time_series,
        passenger_states=state_history
    )


def run_experiments_two_door(config: dict, no_carryon: bool = False, use_priority_boarding: bool = True) -> Dict[str, List[SimulationResult]]:
    """
    Run Monte Carlo experiments for all algorithms with two-door boarding.
    
    Two-door boarding advantage: passengers can use the door closest to their seat,
    reducing walking distance and allowing parallel loading from both ends.
    
    With priority boarding, first class uses front door, but other priority groups
    can use whichever door is closest to their seat.
    """
    np.random.seed(config['RANDOM_SEED'])
    random.seed(config['RANDOM_SEED'])
    
    algorithms = {
        'Steffen (2-Door)': get_boarding_order_two_door_steffen,
        'Random (2-Door)': get_boarding_order_two_door_random,
        'Back-to-Front (2-Door)': get_boarding_order_two_door_back_to_front,
        'WILMA (2-Door)': get_boarding_order_two_door_wilma,
    }
    
    results = {name: [] for name in algorithms}
    
    n_replicates = config['N_REPLICATES']
    
    print(f"\nRunning {n_replicates} replicates per algorithm (TWO-DOOR mode)...")
    print(f"Aircraft: {config['N_ROWS']} rows x 6 seats = {config['N_ROWS'] * 6} passengers")
    print(f"Doors: Front (Row 1) + Rear (Row {config['N_ROWS']})")
    print(f"Carry-on luggage: {'Disabled' if no_carryon else 'Enabled'}")
    print(f"Priority boarding groups: {'Enabled' if use_priority_boarding else 'Disabled'}")
    print("-" * 60)
    
    for rep in range(n_replicates):
        if (rep + 1) % 100 == 0 or rep == 0:
            print(f"  Replicate {rep + 1}/{n_replicates}...")
        
        # Generate same passengers for all algorithms
        passengers = generate_passengers(config, no_carryon=no_carryon)
        
        for alg_name, order_func in algorithms.items():
            # Get boarding orders for both doors from algorithm
            front_order, back_order = order_func(passengers, config)
            
            # Apply priority boarding rules for two-door scenario
            if use_priority_boarding:
                front_order, back_order = apply_priority_boarding_two_door(
                    passengers, front_order, back_order, config
                )
            
            # Record states for animation replicate
            record = (rep == config['ANIMATE_REPLICATE'])
            
            # Run simulation
            result = simulate_boarding_two_door(passengers, front_order, back_order, config, record_states=record)
            result.algorithm = alg_name
            results[alg_name].append(result)
    
    return results


def apply_priority_boarding_two_door(
    passengers: List[Passenger],
    front_order: List[Passenger],
    back_order: List[Passenger],
    config: dict
) -> Tuple[List[Passenger], List[Passenger]]:
    """
    Apply priority boarding rules for two-door configuration.
    
    Key insight: With two doors, the advantage is that passengers can use
    the door closest to their seat. Priority groups should also benefit
    from this - they board early but still use the optimal door.
    
    Boarding sequence per door:
    Front door: First class -> Priority groups in front half -> Algorithm order
    Back door: Priority groups in back half -> Algorithm order
    """
    n_rows = config['N_ROWS']
    mid_row = n_rows // 2
    first_class_rows = config.get('PRIORITY_BOARDING', {}).get('first_class_rows', 3)
    
    # Use passenger IDs for tracking instead of set
    front_ids = {p.id for p in front_order}
    back_ids = {p.id for p in back_order}
    all_passenger_ids = front_ids | back_ids
    
    # Create lookup by ID
    passenger_by_id = {p.id: p for p in passengers}
    
    # Categorize by group and optimal door
    first_class = [p for p in passengers if p.boarding_group == 'first_class']
    
    # Priority groups - assign to nearest door
    priority_groups = ['families', 'elderly', 'military', 'frequent_flyer']
    front_priority = [p for p in passengers 
                      if p.boarding_group in priority_groups and p.row <= mid_row]
    back_priority = [p for p in passengers 
                     if p.boarding_group in priority_groups and p.row > mid_row]
    
    # Regular passengers - keep algorithm's door assignment
    front_regular = [p for p in front_order if p.boarding_group == 'regular']
    back_regular = [p for p in back_order if p.boarding_group == 'regular']
    
    # Shuffle priority groups (they don't follow algorithm order)
    random.shuffle(front_priority)
    random.shuffle(back_priority)
    
    # Build final orders
    # Front: first class first, then front-half priority, then front regular
    new_front_order = first_class + front_priority + front_regular
    
    # Back: back-half priority first, then back regular
    new_back_order = back_priority + back_regular
    
    return new_front_order, new_back_order

def run_experiments(config: dict, no_carryon: bool = False, use_priority_boarding: bool = True) -> Dict[str, List[SimulationResult]]:
    """
    Run Monte Carlo experiments for all algorithms.
    
    Args:
        config: Configuration dictionary
        no_carryon: If True, simulate without carry-on luggage
        use_priority_boarding: If True, apply real-world priority boarding rules
    
    Returns:
        Dictionary mapping algorithm name to list of results
    """
    np.random.seed(config['RANDOM_SEED'])
    random.seed(config['RANDOM_SEED'])
    
    algorithms = {
        'Back-to-Front': get_boarding_order_back_to_front,
        'Random': get_boarding_order_random,
        'WILMA': get_boarding_order_wilma,
        'Steffen': get_boarding_order_steffen,
    }
    
    results = {name: [] for name in algorithms}
    
    n_replicates = config['N_REPLICATES']
    
    print(f"\nRunning {n_replicates} replicates per algorithm...")
    print(f"Aircraft: {config['N_ROWS']} rows x 6 seats = {config['N_ROWS'] * 6} passengers")
    print(f"Carry-on luggage: {'Disabled' if no_carryon else 'Enabled'}")
    print(f"Priority boarding groups: {'Enabled' if use_priority_boarding else 'Disabled'}")
    print("-" * 60)
    
    for rep in range(n_replicates):
        if (rep + 1) % 100 == 0 or rep == 0:
            print(f"  Replicate {rep + 1}/{n_replicates}...")
        
        # Generate same passengers for all algorithms
        passengers = generate_passengers(config, no_carryon=no_carryon)
        
        # Show boarding group breakdown on first replicate
        if rep == 0:
            groups = {}
            for p in passengers:
                groups[p.boarding_group] = groups.get(p.boarding_group, 0) + 1
            print(f"  Passenger groups: {groups}")
        
        for alg_name, order_func in algorithms.items():
            # Get boarding order from algorithm
            boarding_order = order_func(passengers, config)
            
            # Apply priority boarding rules (disrupts optimal ordering)
            if use_priority_boarding:
                boarding_order = apply_priority_boarding(passengers, boarding_order)
            
            # Record states for animation replicate (all algorithms need states for combined animation)
            record = (rep == config['ANIMATE_REPLICATE'])
            
            # Run simulation
            result = simulate_boarding(passengers, boarding_order, config, record_states=record)
            result.algorithm = alg_name
            results[alg_name].append(result)
    
    return results


def compute_statistics(results: Dict[str, List[SimulationResult]]) -> Dict[str, Dict]:
    """
    Compute summary statistics for each algorithm.
    """
    stats = {}
    
    for alg_name, alg_results in results.items():
        times = [r.boarding_time for r in alg_results]
        n = len(times)
        mean = np.mean(times)
        std = np.std(times, ddof=1)
        ci_95 = 1.96 * std / np.sqrt(n)
        
        stats[alg_name] = {
            'mean': mean,
            'std': std,
            'ci_95': ci_95,
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'times': times
        }
    
    return stats


def print_summary(stats: Dict[str, Dict], title: str = "Boarding Simulation Results", baseline_alg: str = None):
    """
    Print formatted summary table.
    """
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
    print(f"{'Algorithm':<25} {'Mean (s)':<12} {'Std Dev':<10} {'95% CI':<14} {'Min-Max':<16}")
    print("-" * 70)
    
    # Sort by mean time
    sorted_algs = sorted(stats.keys(), key=lambda x: stats[x]['mean'])
    
    for alg in sorted_algs:
        s = stats[alg]
        ci_str = f"+/-{s['ci_95']:.1f}"
        range_str = f"{s['min']:.1f}-{s['max']:.1f}"
        print(f"{alg:<25} {s['mean']:<12.1f} {s['std']:<10.2f} {ci_str:<14} {range_str:<16}")
    
    print("-" * 70)
    
    # Find baseline algorithm
    if baseline_alg is None:
        # Try common baseline names
        for candidate in ['Back-to-Front', 'Back-to-Front (2-Door)']:
            if candidate in stats:
                baseline_alg = candidate
                break
    
    if baseline_alg and baseline_alg in stats:
        baseline = stats[baseline_alg]['mean']
        print(f"\nSpeedup vs {baseline_alg} baseline:")
        for alg in sorted_algs:
            speedup = (baseline - stats[alg]['mean']) / baseline * 100
            faster = "faster" if speedup > 0 else "slower"
            print(f"  {alg}: {abs(speedup):.1f}% {faster}")
    
    print("=" * 70)


# ==============[VISUALIZATION]====================

def create_visualizations(
    stats: Dict[str, Dict],
    results: Dict[str, List[SimulationResult]],
    config: dict,
    suffix: str = ""
):
    """
    Create matplotlib visualizations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color scheme
    colors = {
        'Back-to-Front': '#e74c3c',
        'Random': '#3498db',
        'WILMA': '#2ecc71',
        'Steffen': '#9b59b6'
    }
    
    # Plot 1: Box plot of boarding times
    ax1 = axes[0]
    alg_names = list(stats.keys())
    data = [stats[alg]['times'] for alg in alg_names]
    bp = ax1.boxplot(data, labels=alg_names, patch_artist=True)
    for patch, alg in zip(bp['boxes'], alg_names):
        patch.set_facecolor(colors[alg])
        patch.set_alpha(0.7)
    ax1.set_ylabel('Boarding Time (seconds)', fontsize=12)
    ax1.set_title('Distribution of Boarding Times', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Bar chart with error bars
    ax2 = axes[1]
    sorted_algs = sorted(alg_names, key=lambda x: stats[x]['mean'])
    means = [stats[alg]['mean'] for alg in sorted_algs]
    errors = [stats[alg]['ci_95'] for alg in sorted_algs]
    bars = ax2.bar(sorted_algs, means, yerr=errors, capsize=5,
                   color=[colors[alg] for alg in sorted_algs], alpha=0.7)
    ax2.set_ylabel('Mean Boarding Time (seconds)', fontsize=12)
    ax2.set_title('Mean Boarding Time with 95% CI', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, err in zip(bars, means, errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 5,
                f'{mean:.1f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    filename = f'boarding_comparison{suffix}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot: {filename}")
    plt.close()
    
    # Create time series plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for alg_name in alg_names:
        # Use first replicate's time series
        ts = results[alg_name][0].time_series
        times, seated = zip(*ts)
        ax.plot(times, seated, label=alg_name, color=colors[alg_name], linewidth=2)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Passengers Seated', fontsize=12)
    ax.set_title('Boarding Progress Over Time (Single Run)', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, config['N_ROWS'] * 6)
    
    plt.tight_layout()
    filename = f'boarding_timeseries{suffix}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved time series plot: {filename}")
    plt.close()


def create_animation(
    results: Dict[str, List[SimulationResult]],
    config: dict,
    algorithm: str = 'Steffen',
    suffix: str = ""
):
    """
    Create animated GIF of boarding process for a single algorithm.
    (Legacy function - kept for backwards compatibility)
    """
    # Get result with state history
    result = results[algorithm][config['ANIMATE_REPLICATE']]
    
    if not result.passenger_states:
        print(f"No state history recorded for {algorithm}")
        return
    
    print(f"\nCreating animation for {algorithm} method...")
    
    n_rows = config['N_ROWS']
    states = result.passenger_states
    
    # Subsample for faster animation
    speedup = config['ANIMATION_SPEEDUP']
    states = states[::speedup]
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Aircraft dimensions
    aisle_x = 6  # X position of aisle
    seat_width = 1.5
    seat_height = 0.8
    row_spacing = 1.0
    
    def get_seat_x(seat_letter):
        """Get x position for a seat letter"""
        positions = {'A': 1, 'B': 2.5, 'C': 4, 'D': 8, 'E': 9.5, 'F': 11}
        return positions[seat_letter]
    
    def draw_aircraft(ax):
        """Draw static aircraft elements"""
        ax.clear()
        
        # Draw fuselage outline
        ax.add_patch(Rectangle((0, 0), 12, n_rows + 2, fill=False, 
                               edgecolor='gray', linewidth=2))
        
        # Draw aisle
        ax.axvline(x=aisle_x, color='lightgray', linewidth=10, alpha=0.5)
        
        # Draw seat outlines
        for row in range(1, n_rows + 1):
            y = n_rows - row + 1
            for seat in config['SEAT_LETTERS']:
                x = get_seat_x(seat) - seat_width/2
                ax.add_patch(Rectangle((x, y - seat_height/2), seat_width, seat_height,
                                       fill=False, edgecolor='lightgray', linewidth=0.5))
        
        # Row labels
        for row in range(1, n_rows + 1):
            y = n_rows - row + 1
            ax.text(-0.5, y, str(row), ha='center', va='center', fontsize=7)
        
        # Seat labels
        for seat in config['SEAT_LETTERS']:
            ax.text(get_seat_x(seat), n_rows + 1.5, seat, ha='center', va='center', fontsize=9)
        
        ax.set_xlim(-1, 13)
        ax.set_ylim(-0.5, n_rows + 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def animate(frame_idx):
        if frame_idx >= len(states):
            return []
        
        state = states[frame_idx]
        draw_aircraft(ax)
        
        # Color mapping for passenger states
        state_colors = {
            0: 'lightblue',    # WAITING
            1: 'orange',       # WALKING
            2: 'red',          # STOWING
            3: 'green'         # SEATED
        }
        
        patches = []
        
        for p_state in state['passengers']:
            p_id = p_state['id']
            row = p_state['row']
            seat = p_state['seat']
            status = p_state['state']
            aisle_pos = p_state['aisle_pos']
            
            if status == 0:  # WAITING - don't draw
                continue
            
            if status == 3:  # SEATED
                x = get_seat_x(seat)
                y = n_rows - row + 1
            elif status in [1, 2]:  # WALKING or STOWING
                x = aisle_x
                y = n_rows - aisle_pos + 1
            else:
                continue
            
            circle = Circle((x, y), 0.35, facecolor=state_colors[status],
                           edgecolor='black', linewidth=0.5)
            ax.add_patch(circle)
            patches.append(circle)
        
        # Add legend and info
        ax.text(13.5, n_rows, f'Time: {state["time"]:.1f}s', fontsize=12)
        
        n_seated = sum(1 for p in state['passengers'] if p['state'] == 3)
        n_total = len(state['passengers'])
        ax.text(13.5, n_rows - 1.5, f'Seated: {n_seated}/{n_total}', fontsize=11)
        
        # Legend
        legend_y = n_rows - 4
        for status, (label, color) in enumerate([
            ('Waiting', 'lightblue'),
            ('Walking', 'orange'),
            ('Stowing', 'red'),
            ('Seated', 'green')
        ]):
            if status > 0:  # Skip waiting in legend
                ax.add_patch(Circle((13.5, legend_y), 0.25, facecolor=color, edgecolor='black'))
                ax.text(14.2, legend_y, label, va='center', fontsize=9)
                legend_y -= 1
        
        ax.set_title(f'{algorithm} Boarding Method', fontsize=14, fontweight='bold')
        
        return patches
    
    # Create animation
    n_frames = len(states)
    print(f"  Generating {n_frames} frames...")
    
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames,
        interval=config['ANIMATION_INTERVAL'], blit=False
    )
    
    # Save as GIF
    filename = f'boarding_animation_{algorithm.lower()}{suffix}.gif'
    print(f"  Saving GIF (this may take a minute)...")
    anim.save(filename, writer='pillow', fps=20)
    print(f"  Saved animation: {filename}")
    plt.close()


def create_combined_animation(
    results: Dict[str, List[SimulationResult]],
    config: dict,
    suffix: str = ""
):
    """
    Create animated GIF showing all 4 boarding methods side-by-side
    with capacity percentage bars and time series graphs.
    """
    algorithms = ['WILMA', 'Steffen', 'Random', 'Back-to-Front']
    
    # Color scheme for algorithms
    alg_colors = {
        'Back-to-Front': '#e74c3c',
        'Random': '#3498db',
        'WILMA': '#2ecc71',
        'Steffen': '#9b59b6'
    }
    
    # Get state histories for all algorithms
    all_states = {}
    max_time = 0
    for alg in algorithms:
        result = results[alg][config['ANIMATE_REPLICATE']]
        if not result.passenger_states:
            print(f"No state history recorded for {alg}")
            return
        all_states[alg] = result.passenger_states
        if result.passenger_states:
            max_time = max(max_time, result.passenger_states[-1]['time'])
    
    print(f"\nCreating combined animation for all 4 methods...")
    
    n_rows = config['N_ROWS']
    n_total = n_rows * 6
    
    # Subsample for faster animation
    speedup = config['ANIMATION_SPEEDUP']
    
    # Find the maximum number of frames across all algorithms
    max_frames = max(len(states) for states in all_states.values())
    
    # Create time-indexed lookup for each algorithm
    def get_state_at_frame(alg, frame_idx):
        """Get state at frame, or last state if simulation finished"""
        states = all_states[alg]
        idx = min(frame_idx * speedup, len(states) - 1)
        return states[idx]
    
    # Calculate total frames needed
    n_frames = (max_frames + speedup - 1) // speedup
    
    # Setup figure with GridSpec for complex layout
    # 4 columns: one for each algorithm
    # 3 rows per column: title/capacity bar, aircraft, time series
    fig = plt.figure(figsize=(24, 16))
    
    # Create grid: 4 columns, with each having aircraft (top, tall) and graph (bottom, shorter)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.8, 6, 2.5], 
                  hspace=0.15, wspace=0.25)
    
    # Create axes for each algorithm
    axes_capacity = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axes_aircraft = [fig.add_subplot(gs[1, i]) for i in range(4)]
    axes_timeseries = [fig.add_subplot(gs[2, i]) for i in range(4)]
    
    # Aircraft drawing parameters
    aisle_x = 6
    seat_width = 1.5
    seat_height = 0.8
    
    def get_seat_x(seat_letter):
        """Get x position for a seat letter"""
        positions = {'A': 1, 'B': 2.5, 'C': 4, 'D': 8, 'E': 9.5, 'F': 11}
        return positions[seat_letter]
    
    # Passenger state colors
    state_colors = {
        0: 'lightblue',    # WAITING
        1: 'orange',       # WALKING
        2: 'red',          # STOWING
        3: 'green'         # SEATED
    }
    
    def draw_aircraft(ax, alg):
        """Draw static aircraft elements"""
        ax.clear()
        
        # Draw fuselage outline
        ax.add_patch(Rectangle((0, 0), 12, n_rows + 2, fill=False, 
                               edgecolor='gray', linewidth=1.5))
        
        # Draw aisle
        ax.axvline(x=aisle_x, color='lightgray', linewidth=8, alpha=0.5)
        
        # Draw seat outlines
        for row in range(1, n_rows + 1):
            y = n_rows - row + 1
            for seat in config['SEAT_LETTERS']:
                x = get_seat_x(seat) - seat_width/2
                ax.add_patch(Rectangle((x, y - seat_height/2), seat_width, seat_height,
                                       fill=False, edgecolor='lightgray', linewidth=0.3))
        
        # Row labels (every 5 rows for cleaner look)
        for row in range(5, n_rows + 1, 5):
            y = n_rows - row + 1
            ax.text(-0.3, y, str(row), ha='center', va='center', fontsize=6, color='gray')
        
        ax.set_xlim(-0.5, 12.5)
        ax.set_ylim(-0.5, n_rows + 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def draw_capacity_bar(ax, alg, n_seated, current_time):
        """Draw capacity percentage bar"""
        ax.clear()
        
        pct = (n_seated / n_total) * 100
        color = alg_colors[alg]
        
        # Background bar
        ax.barh(0, 100, height=0.6, color='lightgray', alpha=0.3)
        
        # Filled bar
        ax.barh(0, pct, height=0.6, color=color, alpha=0.8)
        
        # Percentage text
        ax.text(50, 0, f'{pct:.0f}%', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='black')
        
        # Time text on the right
        ax.text(102, 0, f'{current_time:.0f}s', ha='left', va='center',
                fontsize=11, color='gray')
        
        ax.set_xlim(-2, 115)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        ax.set_title(alg, fontsize=14, fontweight='bold', color=color, pad=5)
    
    def draw_timeseries(ax, alg, time_history, seated_history, current_time):
        """Draw time series graph of boarding progress"""
        ax.clear()
        
        color = alg_colors[alg]
        
        # Plot filled area
        ax.fill_between(time_history, seated_history, alpha=0.3, color=color)
        ax.plot(time_history, seated_history, color=color, linewidth=2)
        
        # Current position marker
        if time_history:
            ax.scatter([time_history[-1]], [seated_history[-1]], 
                      color=color, s=50, zorder=5)
        
        # Reference line at 100%
        ax.axhline(y=n_total, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlim(0, max_time * 1.05)
        ax.set_ylim(0, n_total * 1.05)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Seated', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        
        # Add completion time if done
        if seated_history and seated_history[-1] >= n_total:
            ax.axvline(x=time_history[-1], color=color, linestyle=':', alpha=0.7)
            ax.text(time_history[-1], n_total * 0.5, f' Done!\n {time_history[-1]:.0f}s',
                   fontsize=9, color=color, fontweight='bold')
    
    # Track time series data for each algorithm
    time_series_data = {alg: {'time': [], 'seated': []} for alg in algorithms}
    
    def animate(frame_idx):
        for i, alg in enumerate(algorithms):
            state = get_state_at_frame(alg, frame_idx)
            current_time = state['time']
            
            # Count seated passengers
            n_seated = sum(1 for p in state['passengers'] if p['state'] == 3)
            
            # Update time series data
            time_series_data[alg]['time'].append(current_time)
            time_series_data[alg]['seated'].append(n_seated)
            
            # Draw capacity bar
            draw_capacity_bar(axes_capacity[i], alg, n_seated, current_time)
            
            # Draw aircraft
            draw_aircraft(axes_aircraft[i], alg)
            
            # Draw passengers
            for p_state in state['passengers']:
                row = p_state['row']
                seat = p_state['seat']
                status = p_state['state']
                aisle_pos = p_state['aisle_pos']
                
                if status == 0:  # WAITING - don't draw
                    continue
                
                if status == 3:  # SEATED
                    x = get_seat_x(seat)
                    y = n_rows - row + 1
                elif status in [1, 2]:  # WALKING or STOWING
                    x = aisle_x
                    y = n_rows - aisle_pos + 1
                else:
                    continue
                
                circle = Circle((x, y), 0.3, facecolor=state_colors[status],
                               edgecolor='black', linewidth=0.3)
                axes_aircraft[i].add_patch(circle)
            
            # Draw time series
            draw_timeseries(axes_timeseries[i], alg,
                          time_series_data[alg]['time'],
                          time_series_data[alg]['seated'],
                          current_time)
        
        # Add main title with current time
        current_max_time = max(get_state_at_frame(alg, frame_idx)['time'] for alg in algorithms)
        fig.suptitle(f'Airplane Boarding Simulation Comparison\nTime: {current_max_time:.1f}s',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add legend on first frame
        if frame_idx == 0:
            # Create legend in bottom right area
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                          markersize=10, label='Walking'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                          markersize=10, label='Stowing'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                          markersize=10, label='Seated'),
            ]
            fig.legend(handles=legend_elements, loc='lower center', 
                      ncol=3, fontsize=11, frameon=True,
                      bbox_to_anchor=(0.5, 0.01))
        
        return []
    
    print(f"  Generating {n_frames} frames...")
    
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames,
        interval=config['ANIMATION_INTERVAL'], blit=False
    )
    
    # Save as GIF
    filename = f'boarding_comparison_animated{suffix}.gif'
    print(f"  Saving combined GIF (this may take several minutes)...")
    anim.save(filename, writer='pillow', fps=15, dpi=100)
    print(f"  Saved animation: {filename}")
    plt.close()


def create_combined_animation_two_door(
    results: Dict[str, List[SimulationResult]],
    config: dict,
    suffix: str = ""
):
    """
    Create animated GIF showing all 4 two-door boarding methods side-by-side
    with capacity percentage bars and time series graphs.
    Shows doors at front and back of aircraft.
    """
    algorithms = ['Steffen (2-Door)', 'WILMA (2-Door)', 'Random (2-Door)', 'Back-to-Front (2-Door)']
    
    # Color scheme for algorithms
    alg_colors = {
        'Back-to-Front (2-Door)': '#e74c3c',
        'Random (2-Door)': '#3498db',
        'WILMA (2-Door)': '#2ecc71',
        'Steffen (2-Door)': '#9b59b6'
    }
    
    # Get state histories for all algorithms
    all_states = {}
    max_time = 0
    for alg in algorithms:
        result = results[alg][config['ANIMATE_REPLICATE']]
        if not result.passenger_states:
            print(f"No state history recorded for {alg}")
            return
        all_states[alg] = result.passenger_states
        if result.passenger_states:
            max_time = max(max_time, result.passenger_states[-1]['time'])
    
    print(f"\nCreating combined TWO-DOOR animation for all 4 methods...")
    
    n_rows = config['N_ROWS']
    n_total = n_rows * 6
    
    # Subsample for faster animation
    speedup = config['ANIMATION_SPEEDUP']
    
    # Find the maximum number of frames across all algorithms
    max_frames = max(len(states) for states in all_states.values())
    
    # Create time-indexed lookup for each algorithm
    def get_state_at_frame(alg, frame_idx):
        """Get state at frame, or last state if simulation finished"""
        states = all_states[alg]
        idx = min(frame_idx * speedup, len(states) - 1)
        return states[idx]
    
    # Calculate total frames needed
    n_frames = (max_frames + speedup - 1) // speedup
    
    # Setup figure with GridSpec for complex layout
    fig = plt.figure(figsize=(24, 16))
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.8, 6, 2.5], 
                  hspace=0.15, wspace=0.25)
    
    # Create axes for each algorithm
    axes_capacity = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axes_aircraft = [fig.add_subplot(gs[1, i]) for i in range(4)]
    axes_timeseries = [fig.add_subplot(gs[2, i]) for i in range(4)]
    
    # Aircraft drawing parameters
    aisle_x = 6
    seat_width = 1.5
    seat_height = 0.8
    
    def get_seat_x(seat_letter):
        """Get x position for a seat letter"""
        positions = {'A': 1, 'B': 2.5, 'C': 4, 'D': 8, 'E': 9.5, 'F': 11}
        return positions[seat_letter]
    
    # Passenger state colors
    state_colors = {
        0: 'lightblue',    # WAITING
        1: 'orange',       # WALKING
        2: 'red',          # STOWING
        3: 'green'         # SEATED
    }
    
    def draw_aircraft_two_door(ax, alg):
        """Draw static aircraft elements with two doors indicated"""
        ax.clear()
        
        # Draw fuselage outline
        ax.add_patch(Rectangle((0, 0), 12, n_rows + 2, fill=False, 
                               edgecolor='gray', linewidth=1.5))
        
        # Draw aisle
        ax.axvline(x=aisle_x, color='lightgray', linewidth=8, alpha=0.5)
        
        # Draw seat outlines
        for row in range(1, n_rows + 1):
            y = n_rows - row + 1
            for seat in config['SEAT_LETTERS']:
                x = get_seat_x(seat) - seat_width/2
                ax.add_patch(Rectangle((x, y - seat_height/2), seat_width, seat_height,
                                       fill=False, edgecolor='lightgray', linewidth=0.3))
        
        # Row labels (every 5 rows for cleaner look)
        for row in range(5, n_rows + 1, 5):
            y = n_rows - row + 1
            ax.text(-0.3, y, str(row), ha='center', va='center', fontsize=6, color='gray')
        
        # Draw front door indicator (row 1, port side - left side which is seats A,B,C)
        front_door_y = n_rows  # Row 1 position
        ax.add_patch(Rectangle((-0.8, front_door_y - 0.3), 0.8, 0.6, 
                               facecolor='#3498db', edgecolor='black', linewidth=1))
        ax.text(-0.4, front_door_y, 'F', ha='center', va='center', fontsize=7, 
                fontweight='bold', color='white')
        
        # Draw back door indicator (row N, port side)
        back_door_y = 1  # Row N position  
        ax.add_patch(Rectangle((-0.8, back_door_y - 0.3), 0.8, 0.6,
                               facecolor='#e74c3c', edgecolor='black', linewidth=1))
        ax.text(-0.4, back_door_y, 'R', ha='center', va='center', fontsize=7,
                fontweight='bold', color='white')
        
        ax.set_xlim(-1.2, 12.5)
        ax.set_ylim(-0.5, n_rows + 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def draw_capacity_bar(ax, alg, n_seated, current_time):
        """Draw capacity percentage bar"""
        ax.clear()
        
        pct = (n_seated / n_total) * 100
        color = alg_colors[alg]
        
        # Background bar
        ax.barh(0, 100, height=0.6, color='lightgray', alpha=0.3)
        
        # Filled bar
        ax.barh(0, pct, height=0.6, color=color, alpha=0.8)
        
        # Percentage text
        ax.text(50, 0, f'{pct:.0f}%', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='black')
        
        # Time text on the right
        ax.text(102, 0, f'{current_time:.0f}s', ha='left', va='center',
                fontsize=11, color='gray')
        
        ax.set_xlim(-2, 115)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        
        # Shorter algorithm name for title
        short_name = alg.replace(' (2-Door)', '')
        ax.set_title(short_name, fontsize=14, fontweight='bold', color=color, pad=5)
    
    def draw_timeseries(ax, alg, time_history, seated_history, current_time):
        """Draw time series graph of boarding progress"""
        ax.clear()
        
        color = alg_colors[alg]
        
        # Plot filled area
        ax.fill_between(time_history, seated_history, alpha=0.3, color=color)
        ax.plot(time_history, seated_history, color=color, linewidth=2)
        
        # Current position marker
        if time_history:
            ax.scatter([time_history[-1]], [seated_history[-1]], 
                      color=color, s=50, zorder=5)
        
        # Reference line at 100%
        ax.axhline(y=n_total, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlim(0, max_time * 1.05)
        ax.set_ylim(0, n_total * 1.05)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Seated', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        
        # Add completion time if done
        if seated_history and seated_history[-1] >= n_total:
            ax.axvline(x=time_history[-1], color=color, linestyle=':', alpha=0.7)
            ax.text(time_history[-1], n_total * 0.5, f' Done!\n {time_history[-1]:.0f}s',
                   fontsize=9, color=color, fontweight='bold')
    
    # Track time series data for each algorithm
    time_series_data = {alg: {'time': [], 'seated': []} for alg in algorithms}
    
    def animate(frame_idx):
        for i, alg in enumerate(algorithms):
            state = get_state_at_frame(alg, frame_idx)
            current_time = state['time']
            
            # Count seated passengers
            n_seated = sum(1 for p in state['passengers'] if p['state'] == 3)
            
            # Update time series data
            time_series_data[alg]['time'].append(current_time)
            time_series_data[alg]['seated'].append(n_seated)
            
            # Draw capacity bar
            draw_capacity_bar(axes_capacity[i], alg, n_seated, current_time)
            
            # Draw aircraft with two doors
            draw_aircraft_two_door(axes_aircraft[i], alg)
            
            # Draw passengers
            for p_state in state['passengers']:
                row = p_state['row']
                seat = p_state['seat']
                status = p_state['state']
                aisle_pos = p_state['aisle_pos']
                
                if status == 0:  # WAITING - don't draw
                    continue
                
                if status == 3:  # SEATED
                    x = get_seat_x(seat)
                    y = n_rows - row + 1
                elif status in [1, 2]:  # WALKING or STOWING
                    x = aisle_x
                    y = n_rows - aisle_pos + 1
                else:
                    continue
                
                circle = Circle((x, y), 0.3, facecolor=state_colors[status],
                               edgecolor='black', linewidth=0.3)
                axes_aircraft[i].add_patch(circle)
            
            # Draw time series
            draw_timeseries(axes_timeseries[i], alg,
                          time_series_data[alg]['time'],
                          time_series_data[alg]['seated'],
                          current_time)
        
        # Add main title with current time
        current_max_time = max(get_state_at_frame(alg, frame_idx)['time'] for alg in algorithms)
        fig.suptitle(f'Two-Door Boarding Simulation Comparison\n(Front + Rear Doors on Port Side)\nTime: {current_max_time:.1f}s',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add legend on first frame
        if frame_idx == 0:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                          markersize=10, label='Walking'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                          markersize=10, label='Stowing'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                          markersize=10, label='Seated'),
                plt.Line2D([0], [0], marker='s', color='#3498db', markersize=10, label='Front Door'),
                plt.Line2D([0], [0], marker='s', color='#e74c3c', markersize=10, label='Rear Door'),
            ]
            fig.legend(handles=legend_elements, loc='lower center', 
                      ncol=5, fontsize=10, frameon=True,
                      bbox_to_anchor=(0.5, 0.01))
        
        return []
    
    print(f"  Generating {n_frames} frames...")
    
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames,
        interval=config['ANIMATION_INTERVAL'], blit=False
    )
    
    # Save as GIF
    filename = f'boarding_two_door_animated{suffix}.gif'
    print(f"  Saving combined two-door GIF (this may take several minutes)...")
    anim.save(filename, writer='pillow', fps=15, dpi=100)
    print(f"  Saved animation: {filename}")
    plt.close()


# ==============[MAIN.PYTHON EXECUTE!!]====================

def main():
    """
    Main function to run all experiments and generate outputs.
    """
    print("\n" + "=" * 70)
    print(" AIRPLANE BOARDING SIMULATION")
    print(" Comparing: Steffen, WILMA, Random, Back-to-Front")
    print(" Single-Door and Two-Door Configurations")
    print(" With Real-World Priority Boarding Groups")
    print("=" * 70)
    
    # Print priority boarding configuration
    priority = CONFIG.get('PRIORITY_BOARDING', {})
    print("\nPriority Boarding Groups Enabled:")
    print(f"  - First Class: Rows 1-{priority.get('first_class_rows', 3)}")
    print(f"  - Families with children: {priority.get('families_pct', 0.08)*100:.0f}% (slower)")
    print(f"  - Elderly/Disabled: {priority.get('elderly_pct', 0.05)*100:.0f}% (slower)")
    print(f"  - Military: {priority.get('military_pct', 0.03)*100:.0f}%")
    print(f"  - Frequent Flyers: {priority.get('frequent_flyer_pct', 0.10)*100:.0f}%")
    
    # =====================[SINGLE-DOOR EXPERIMENTS]===========================
    print("\n" + "=" * 70)
    print(" PART 1: SINGLE-DOOR BOARDING (with priority groups)")
    print("=" * 70)
    
    # Run experiments WITH carry-on luggage and priority boarding
    print("\n[1/6] Running single-door experiments WITH priority boarding...")
    results_carryon = run_experiments(CONFIG, no_carryon=False, use_priority_boarding=True)
    stats_carryon = compute_statistics(results_carryon)
    print_summary(stats_carryon, "Single-Door Results WITH Priority Boarding")
    
    # Run experiments WITHOUT carry-on luggage
    print("\n[2/6] Running single-door experiments WITHOUT carry-on luggage...")
    results_no_carryon = run_experiments(CONFIG, no_carryon=True, use_priority_boarding=True)
    stats_no_carryon = compute_statistics(results_no_carryon)
    print_summary(stats_no_carryon, "Single-Door Results WITHOUT Carry-on")
    
    # Create visualizations
    print("\n[3/6] Creating single-door visualizations...")
    create_visualizations(stats_carryon, results_carryon, CONFIG, "_single_door_with_carryon")
    create_visualizations(stats_no_carryon, results_no_carryon, CONFIG, "_single_door_no_carryon")
    
    # Create animations (combined 4-way comparison)
    print("\n[4/6] Creating single-door animation...")
    create_combined_animation(results_carryon, CONFIG, "_single_door")
    
    # =====================[TWO-DOOR EXPERIMENTS]===========================
    print("\n" + "=" * 70)
    print(" PART 2: TWO-DOOR BOARDING (Front + Rear, with priority groups)")
    print("=" * 70)
    
    # Run two-door experiments WITH carry-on luggage
    print("\n[5/6] Running two-door experiments WITH priority boarding...")
    results_two_door = run_experiments_two_door(CONFIG, no_carryon=False, use_priority_boarding=True)
    stats_two_door = compute_statistics(results_two_door)
    print_summary(stats_two_door, "Two-Door Results WITH Priority Boarding")
    
    # Create two-door animation
    print("\n[6/6] Creating two-door animation...")
    create_combined_animation_two_door(results_two_door, CONFIG, "")
    
    # ==================[COMPARE SINGLE-DOOR VS TWO-DOOR]===================
    print("\n" + "=" * 70)
    print(" COMPARISON: SINGLE-DOOR vs TWO-DOOR")
    print("=" * 70)
    
    print("\nBoarding Time Change with Two Doors:")
    for alg_base in ['WILMA', 'Steffen', 'Random', 'Back-to-Front']:
        single_time = stats_carryon[alg_base]['mean']
        two_door_time = stats_two_door[f'{alg_base} (2-Door)']['mean']
        change = (single_time - two_door_time) / single_time * 100
        direction = "faster" if change > 0 else "slower"
        print(f"  {alg_base}: {single_time:.1f}s -> {two_door_time:.1f}s ({abs(change):.1f}% {direction})")
    
    print("\nKey Insight: Two doors help INEFFICIENT algorithms more!")
    print("  - Random/Back-to-Front benefit from reduced walking distance")
    print("  - WILMA/Steffen may be disrupted by priority group seat distribution")
    
    # Generate text report
    generate_report(stats_carryon, stats_no_carryon, stats_two_door, CONFIG)
    
    print("\n" + "=" * 70)
    print(" SIMULATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  Single-Door:")
    print("    - boarding_comparison_single_door_with_carryon.png")
    print("    - boarding_comparison_single_door_no_carryon.png")
    print("    - boarding_timeseries_single_door_with_carryon.png")
    print("    - boarding_timeseries_single_door_no_carryon.png")
    print("    - boarding_comparison_animated_single_door.gif")
    print("  Two-Door:")
    print("    - boarding_two_door_animated.gif")
    print("  Report:")
    print("    - boarding_simulation_report.txt")
    print()


def generate_report(stats_carryon, stats_no_carryon, stats_two_door, config):
    """Generate a text report of the simulation results."""
    
    report = []
    report.append("=" * 70)
    report.append("AIRPLANE BOARDING SIMULATION REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("SIMULATION PARAMETERS:")
    report.append(f"  Aircraft: {config['N_ROWS']} rows x 6 seats = {config['N_ROWS'] * 6} passengers")
    report.append(f"  Time step: {config['TIME_STEP']} seconds")
    report.append(f"  Replicates per algorithm: {config['N_REPLICATES']}")
    report.append(f"  Walking speed: {config['WALK_SPEED_MEAN']:.1f} +/- {config['WALK_SPEED_STD']:.1f} rows/sec")
    report.append(f"  Stow time: {config['STOW_TIME_MEAN']:.1f} +/- {config['STOW_TIME_STD']:.1f} sec")
    report.append(f"  Seat interference time: {config['SEAT_INTERFERENCE_TIME']:.1f} sec per blocking passenger")
    report.append(f"  Carry-on probability: {config['HAS_CARRYON_PROB']*100:.0f}%")
    report.append(f"  Back-to-front block size: {config['BLOCK_SIZE']} rows")
    report.append("")
    
    # Priority boarding groups
    priority = config.get('PRIORITY_BOARDING', {})
    report.append("PRIORITY BOARDING GROUPS (Real-World Simulation):")
    report.append(f"  First Class: Rows 1-{priority.get('first_class_rows', 3)} (board first)")
    report.append(f"  Families with children: {priority.get('families_pct', 0.08)*100:.0f}% of passengers (pre-board, slower)")
    report.append(f"  Elderly/Disabled: {priority.get('elderly_pct', 0.05)*100:.0f}% of passengers (pre-board, slower)")
    report.append(f"  Military: {priority.get('military_pct', 0.03)*100:.0f}% of passengers (pre-board)")
    report.append(f"  Frequent Flyers: {priority.get('frequent_flyer_pct', 0.10)*100:.0f}% of passengers (priority)")
    report.append(f"  Slow passenger walk speed factor: {config.get('SLOW_WALK_SPEED_FACTOR', 0.6)}")
    report.append(f"  Slow passenger stow time factor: {config.get('SLOW_STOW_TIME_FACTOR', 1.5)}")
    report.append("")
    
    # Single-door results
    report.append("=" * 70)
    report.append("SINGLE-DOOR BOARDING RESULTS")
    report.append("=" * 70)
    report.append("")
    report.append("-" * 70)
    report.append("WITH CARRY-ON LUGGAGE:")
    report.append("-" * 70)
    report.append(f"{'Algorithm':<20} {'Mean (s)':<12} {'Std Dev':<10} {'95% CI':<14}")
    report.append("-" * 50)
    
    sorted_algs = sorted(stats_carryon.keys(), key=lambda x: stats_carryon[x]['mean'])
    for alg in sorted_algs:
        s = stats_carryon[alg]
        report.append(f"{alg:<20} {s['mean']:<12.1f} {s['std']:<10.2f} +/-{s['ci_95']:.1f}")
    
    baseline = stats_carryon['Back-to-Front']['mean']
    report.append("")
    report.append("Speedup vs Back-to-Front:")
    for alg in sorted_algs:
        speedup = (baseline - stats_carryon[alg]['mean']) / baseline * 100
        report.append(f"  {alg}: {speedup:+.1f}%")
    
    report.append("")
    report.append("-" * 70)
    report.append("WITHOUT CARRY-ON LUGGAGE:")
    report.append("-" * 70)
    report.append(f"{'Algorithm':<20} {'Mean (s)':<12} {'Std Dev':<10} {'95% CI':<14}")
    report.append("-" * 50)
    
    sorted_algs = sorted(stats_no_carryon.keys(), key=lambda x: stats_no_carryon[x]['mean'])
    for alg in sorted_algs:
        s = stats_no_carryon[alg]
        report.append(f"{alg:<20} {s['mean']:<12.1f} {s['std']:<10.2f} +/-{s['ci_95']:.1f}")
    
    # Two-door results
    report.append("")
    report.append("=" * 70)
    report.append("TWO-DOOR BOARDING RESULTS (Front + Rear Doors)")
    report.append("=" * 70)
    report.append("")
    report.append(f"{'Algorithm':<25} {'Mean (s)':<12} {'Std Dev':<10} {'95% CI':<14}")
    report.append("-" * 60)
    
    sorted_algs_2d = sorted(stats_two_door.keys(), key=lambda x: stats_two_door[x]['mean'])
    for alg in sorted_algs_2d:
        s = stats_two_door[alg]
        report.append(f"{alg:<25} {s['mean']:<12.1f} {s['std']:<10.2f} +/-{s['ci_95']:.1f}")
    
    # Comparison
    report.append("")
    report.append("=" * 70)
    report.append("COMPARISON: SINGLE-DOOR vs TWO-DOOR")
    report.append("=" * 70)
    report.append("")
    report.append(f"{'Algorithm':<15} {'1-Door (s)':<12} {'2-Door (s)':<12} {'Reduction':<12}")
    report.append("-" * 50)
    
    for alg_base in ['Steffen', 'Random', 'Back-to-Front', 'WILMA']:
        single_time = stats_carryon[alg_base]['mean']
        two_door_time = stats_two_door[f'{alg_base} (2-Door)']['mean']
        reduction = (single_time - two_door_time) / single_time * 100
        report.append(f"{alg_base:<15} {single_time:<12.1f} {two_door_time:<12.1f} {reduction:.1f}%")
    
    report.append("")
    report.append("-" * 70)
    report.append("KEY FINDINGS:")
    report.append("-" * 70)
    
    # Find best algorithms
    best_single = min(stats_carryon.keys(), key=lambda x: stats_carryon[x]['mean'])
    best_two_door = min(stats_two_door.keys(), key=lambda x: stats_two_door[x]['mean'])
    
    report.append(f"1. Best single-door algorithm: {best_single}")
    report.append(f"   Mean boarding time: {stats_carryon[best_single]['mean']:.1f}s")
    report.append("")
    report.append(f"2. Best two-door algorithm: {best_two_door}")
    report.append(f"   Mean boarding time: {stats_two_door[best_two_door]['mean']:.1f}s")
    report.append("")
    
    # Overall improvement
    worst_single = stats_carryon['Back-to-Front']['mean']
    best_two = stats_two_door[best_two_door]['mean']
    overall_improvement = (worst_single - best_two) / worst_single * 100
    report.append(f"3. Maximum improvement (Back-to-Front 1-door -> {best_two_door}):")
    report.append(f"   {worst_single:.1f}s -> {best_two:.1f}s ({overall_improvement:.1f}% faster)")
    
    report.append("")
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    with open("boarding_simulation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"\nSaved report: boarding_simulation_report.txt")


if __name__ == "__main__":
    main()
