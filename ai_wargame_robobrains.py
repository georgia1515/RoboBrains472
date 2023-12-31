from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests

import os.path

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4


class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

##############################################################################################################
# GAME RULES


@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount

##############################################################################################################
# BOARD COORDINATES


@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string()+self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row-dist, self.row+1+dist):
            for col in range(self.col-dist, self.col+1+dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1, self.col)
        yield Coord(self.row, self.col-1)
        yield Coord(self.row+1, self.col)
        yield Coord(self.row, self.col+1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################
# GAME MOVES


@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string()+" "+self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row+1):
            for col in range(self.src.col, self.dst.col+1):
                yield Coord(row, col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim-1, dim-1))

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

##############################################################################################################
# GAME STATE


@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = True
    max_turns: int | None = 100
    randomize_moves: bool = True
    broker: str | None = None
    heuristic: int | None = 0   # default heuristic

##############################################################################################################
# GAME STATISTICS


@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    total_time: float = 0.0
    elapsed_time: float = 0.0
    evaluations_per_depth: dict[int, int] = field(default_factory=dict) # to get cumulative evals => sum(evaluations_per_depth.values())
    avg_branching_factor: float = 0.0

##############################################################################################################
# GAME MOVES


@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True
    gameTrace_path: str = ''
    heuristic_score: int = 0

    # number of units for attacker and defender
    numOfProgramsAttacker: int = 2
    numOfFirewallAttacker: int = 1
    numOfProgramsDefender: int = 1
    numOfFirewallDefender: int = 2
    numOfVirusesAttacker: int = 2
    numOfTechsDefender: int = 2
    numOfAIAttacker: int = 1
    numOfAIDefender: int = 1

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim-1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(
            player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(
            player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(
            player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md-1, md),
                 Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md-1),
                 Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md-2, md),
                 Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md-2),
                 Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md-1, md-1),
                 Unit(player=Player.Attacker, type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            # AI
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                    self.numOfAIAttacker = 0
                else:
                    self._defender_has_ai = False
                    self.numOfAIDefender = 0

            # Program
            if unit.type == UnitType.Program:
                if unit.player == Player.Attacker:
                    self.numOfProgramsAttacker -= 1
                else:
                    self.numOfProgramsDefender -= 1

            # Firewall
            if unit.type == UnitType.Firewall:
                if unit.player == Player.Attacker:
                    self.numOfFirewallAttacker -= 1
                else:
                    self.numOfFirewallDefender -= 1

            # Tech
            if unit.type == UnitType.Tech and unit.player == Player.Defender:
                self.numOfTechsDefender -= 1

            # Virus
            if unit.type == UnitType.Virus and unit.player == Player.Attacker:
                self.numOfVirusesAttacker -= 1

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords: CoordPair) -> bool:
        """Validate a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):  # validate if coord inside board game
            return False
        unit = self.get(coords.src)
        if unit is None or unit.player != self.next_player:
            return False

        unitDst = self.get(coords.dst)

        # validate when unit can't move diagonally
        if coords.dst == Coord(coords.src.row-1, coords.src.col-1) or coords.dst == Coord(coords.src.row-1, coords.src.col+1) or coords.dst == Coord(coords.src.row+1, coords.src.col+1) or coords.dst == Coord(coords.src.row+1, coords.src.col-1):
            return False

        # validate when unit engaged in combat with adversial unit -> both unit can't move
        unitAdversarialUp = self.get(
            Coord(coords.src.row-1, coords.src.col))
        unitAdversarialDown = self.get(
            Coord(coords.src.row+1, coords.src.col))
        unitAdversarialLeft = self.get(
            Coord(coords.src.row, coords.src.col-1))
        unitAdversarialRight = self.get(
            Coord(coords.src.row, coords.src.col+1))

        # adversial unit above, below, left, or right of my unit
        # except for type: Virus and Tech -> can move even engaged
        if unit.type == UnitType.AI or unit.type == UnitType.Firewall or unit.type == UnitType.Program:
            if (unitAdversarialUp is not None and unitAdversarialUp.player != unit.player) or (unitAdversarialDown is not None and unitAdversarialDown.player != unit.player) or (unitAdversarialLeft is not None and unitAdversarialLeft.player != unit.player) or (unitAdversarialRight is not None and unitAdversarialRight.player != unit.player):
                return False

        # track unit movement
        deplacementMoveCol = coords.dst.col - coords.src.col
        deplacementMoveRow = coords.dst.row - coords.src.row

        # validate when unit can't move more than 1 place
        if abs(deplacementMoveCol) > 1 or abs(deplacementMoveRow) > 1:
            return False

        # validate attacker move: only 1 left or 1 up
        if unit.player == Player.Attacker and (unit.type == UnitType.AI or unit.type == UnitType.Firewall or unit.type == UnitType.Program):
            if ((deplacementMoveCol == -1 and deplacementMoveRow == 0) or (deplacementMoveCol == 0 and deplacementMoveRow == -1)) and unitDst is None:
                return True
            else:
                return False

        # validate defender move: only 1 right or 1 down
        if unit.player == Player.Defender and (unit.type == UnitType.AI or unit.type == UnitType.Firewall or unit.type == UnitType.Program):
            if ((deplacementMoveCol == 1 and deplacementMoveRow == 0) or (deplacementMoveCol == 0 and deplacementMoveRow == 1)) and unitDst is None:
                return True
            else:
                return False

        unit = self.get(coords.dst)
        return (unit is None)

    def perform_move(self, coords: CoordPair) -> Tuple[bool, str]:
        """Validate and perform a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""

        # validate if my unit can move: if can't, could attack/repair/self-destruct or is invalid move
        if self.is_valid_move(coords):
            self.set(coords.dst, self.get(coords.src))
            self.set(coords.src, None)

            # self.trace_each_action(coords.src, coords.dst)
            return (True, "")

        unit = self.get(coords.src)
        unitDst = self.get(coords.dst)

        if unit is None or unit.player != self.next_player:
            return (False, "invalid move")

        # self-destruct
        if coords.src == coords.dst:
            self.mod_health(coords.src, -9)
            self.mod_health(Coord(coords.src.row-1, coords.src.col), -2)
            self.mod_health(Coord(coords.src.row-1, coords.src.col+1), -2)
            self.mod_health(Coord(coords.src.row, coords.src.col+1), -2)
            self.mod_health(Coord(coords.src.row+1, coords.src.col+1), -2)
            self.mod_health(Coord(coords.src.row+1, coords.src.col), -2)
            self.mod_health(Coord(coords.src.row+1, coords.src.col-1), -2)
            self.mod_health(Coord(coords.src.row, coords.src.col-1), -2)
            self.mod_health(Coord(coords.src.row-1, coords.src.col-1), -2)

            # self.trace_each_action(coords.src, coords.dst)
            return (True, "")

        # get coordinates of adversial unit
        unitAdversarialUp = self.get(
            Coord(coords.src.row-1, coords.src.col))
        unitAdversarialDown = self.get(
            Coord(coords.src.row+1, coords.src.col))
        unitAdversarialRight = self.get(
            Coord(coords.src.row, coords.src.col+1))
        unitAdversarialLeft = self.get(
            Coord(coords.src.row, coords.src.col-1))

        if self.has_attacked_or_repaired(unitAdversarialUp, unit, unitDst, coords) or self.has_attacked_or_repaired(unitAdversarialDown, unit, unitDst, coords) or self.has_attacked_or_repaired(unitAdversarialLeft, unit, unitDst, coords) or self.has_attacked_or_repaired(unitAdversarialRight, unit, unitDst, coords):
            # self.trace_each_action(coords.src, coords.dst)
            return (True, "")

        return (False, "invalid move")

    def has_attacked_or_repaired(self, adjacentUnit, srcUnit, destUnit: Unit, coords: CoordPair) -> bool:
        if adjacentUnit is not None:
            # not same team & dst move is where opponent located at -> attack
            if adjacentUnit.player != srcUnit.player and destUnit == adjacentUnit:
                self.mod_health(
                    coords.src, -(abs(destUnit.damage_amount(srcUnit))))
                self.mod_health(coords.dst, -
                                (abs(srcUnit.damage_amount(destUnit))))
                return True
            # same team & dst move is where friendly unit located at -> repair
            elif adjacentUnit.player == srcUnit.player and destUnit == adjacentUnit and ((srcUnit.type == UnitType.AI and (adjacentUnit.type == UnitType.Virus or adjacentUnit.type == UnitType.Tech)) or (srcUnit.type == UnitType.Tech and (adjacentUnit.type == UnitType.AI or adjacentUnit.type == UnitType.Firewall or adjacentUnit.type == UnitType.Program))):
                if 0 < srcUnit.repair_amount(destUnit) and srcUnit.repair_amount(destUnit) < 9:
                    self.mod_health(
                        coords.dst, +(abs(srcUnit.repair_amount(destUnit))))
                    return True
                else:
                    return False
        return False

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def draw_board(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        coord = Coord()
        board = "   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            board += f"{label:^3} "
        board += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            board += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    board += " .  "
                else:
                    board += f"{str(unit):^3} "
            board += "\n"
        return board

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}/{self.options.max_turns}\n"
        output += "\n"
        output += self.draw_board()
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ", end='')
                    self.trace_each_action(
                        mv.src, mv.dst)
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success, result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ", end='')
                    self.trace_each_action(
                        mv.src, mv.dst)
                    print(result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv)
            if success:
                if self.stats.elapsed_time > self.options.max_time:
                    print(f"Computer {self.next_player.name} has taken longer than the max time allowed!")
                    return None
                else:
                    print(f"Computer {self.next_player.name}: ", end='')
                    self.trace_each_action(mv.src, mv.dst)
                    print(f"")
                    print(result)
                    self.next_turn()
                    return mv
            return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord, unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src, _) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move) or self.is_valid_perform(move): # added is_valid_perform to check if moves are attack/repair/delf-destroy
                    yield move.clone()
            move.dst = src
            yield move.clone()

    # returns true of move is attack/depair/self-destruct
    def is_valid_perform(self, coords: CoordPair) -> bool:
        """simulation: Validate and perform a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""

        # validate if my unit can move: if can't, could attack/repair/self-destruct or is invalid move
        if self.is_valid_move(coords):
            return (True)

        unit = self.get(coords.src)
        unitDst = self.get(coords.dst)

        if unit is None or unit.player != self.next_player:
            return (False)

        # self-destruct
        if coords.src == coords.dst:
            return (True)

        # get coordinates of adversial unit
        unitAdversarialUp = self.get(
            Coord(coords.src.row-1, coords.src.col))
        unitAdversarialDown = self.get(
            Coord(coords.src.row+1, coords.src.col))
        unitAdversarialRight = self.get(
            Coord(coords.src.row, coords.src.col+1))
        unitAdversarialLeft = self.get(
            Coord(coords.src.row, coords.src.col-1))

        if self.can_attack_or_repair(unitAdversarialUp, unit, unitDst, coords) or self.can_attack_or_repair(unitAdversarialDown, unit, unitDst, coords) or self.can_attack_or_repair(unitAdversarialLeft, unit, unitDst, coords) or self.can_attack_or_repair(unitAdversarialRight, unit, unitDst, coords):
            return (True)
        return (False)

    def can_attack_or_repair(self, adjacentUnit, srcUnit, destUnit: Unit, coords: CoordPair) -> bool:
        if adjacentUnit is not None:
            # not same team & dst move is where opponent located at -> attack
            if adjacentUnit.player != srcUnit.player and destUnit == adjacentUnit:
                return True
            # same team & dst move is where friendly unit located at -> repair
            elif adjacentUnit.player == srcUnit.player and destUnit == adjacentUnit and ((srcUnit.type == UnitType.AI and (adjacentUnit.type == UnitType.Virus or adjacentUnit.type == UnitType.Tech)) or (srcUnit.type == UnitType.Tech and (adjacentUnit.type == UnitType.AI or adjacentUnit.type == UnitType.Firewall or adjacentUnit.type == UnitType.Program))):
                if 0 < srcUnit.repair_amount(destUnit) and srcUnit.repair_amount(destUnit) < 9:
                    return True
                else:
                    return False
        return False

    # is not used anymore
    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def suggest_move(self):
        start_time = datetime.now()
        branchingFactor = []
        # will use alpha beta unless options alpha_beta_off
        if (self.options.alpha_beta):
            score, best_move = self.alpha_beta_minimax(
                MIN_HEURISTIC_SCORE, MAX_HEURISTIC_SCORE, self.next_player, 0, branchingFactor, start_time)
        else:
            score, best_move = self.alpha_beta_minimax(None, None, self.next_player, 0, branchingFactor, start_time)
        
        # set stats and score
        self.heuristic_score = score
        self.stats.elapsed_time = (datetime.now() - start_time).total_seconds()
        self.stats.total_time += self.stats.elapsed_time
        # find the average branching factor by finding the sum of all values (number of children) by the number of nodes with children
        self.stats.avg_branching_factor = round(sum(branchingFactor)/len(branchingFactor), 2)
        total_evals = sum(self.stats.evaluations_per_depth.values())
        # prints out information about the chosen move
        print(f"Elapsed time: {self.stats.elapsed_time :0.1f}s")
        print(f"Heuristic score: {self.heuristic_score}")
        print(f"Total evaluations: {total_evals}")
        print(f"Evals per depth: ",end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ",end='')
        print()
        print(f"Cumulative % evals by depth: ", end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}-{self.stats.evaluations_per_depth[k]/sum(self.stats.evaluations_per_depth.values())*100:0.1f}% ",end='')
        print()

        if self.stats.total_time > 0:
            print(f"Eval perf.: {total_evals/self.stats.total_time/1000:0.1f}k/s")
        print(f"Branching factor: {self.stats.avg_branching_factor}")

        # Return the best move
        return best_move
    
    def alpha_beta_minimax(self, alpha, beta, maximizing_player, current_depth, branchingFactor, startTime):
        # base case of the recursive method, recursion ends when we get to the max depth or if game is over
        if self.options.max_depth == current_depth or self.is_finished():
            # increment value by one at the max depth
            self.stats.evaluations_per_depth[current_depth] += 1
            # returns heuristic score based on chosen heuristic
            return self.chosen_heuristic(), None

        best_move = None
        # increment value by one for the evaluations
        self.stats.evaluations_per_depth[current_depth+1] += 1

        # set number of children for this current node
        numOfChildrens = 0

        # maximizing player
        if maximizing_player is Player.Attacker:
            max_score = float("-inf")
            # iterate through all the possible moves
            for move in self.move_candidates():
                # increment value by one for each move (number of children)
                numOfChildrens += 1
                new_game = self.clone()
                new_game.perform_move(move)
                new_game.next_turn()

                # interrupts search depth if not enough time to continue (uncomment the next 3 lines)
                # if (datetime.now() - startTime).total_seconds() >= self.options.max_time -0.5:
                #     print("TOOK TOO LONG")
                #     return max_score, best_move
                
                # calls method recursively with all potential moves after performing move
                score, _, = new_game.alpha_beta_minimax(
                    alpha, beta, Player.Defender, current_depth + 1, branchingFactor, startTime)
                # replaces score if better one found
                if score > max_score:
                    max_score = score
                    best_move = move
                # including alpha beta pruning
                if(alpha is not None and beta is not None):
                    alpha = max(alpha, max_score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
            # add number of children to branching factor list to calculate avg branching factor later
            branchingFactor.append(numOfChildrens)
            return max_score, best_move
        # minimizing player
        else:
            min_score = float("inf")
            # iterate through all the possible moves
            for move in self.move_candidates():
                # increment value by one for each move (number of children)
                numOfChildrens += 1
                new_game = self.clone()
                new_game.perform_move(move)
                new_game.next_turn()

                # interrupts search depth if not enough time to continue (uncomment the next 3 lines)
                # if (datetime.now() - startTime).total_seconds() >= self.options.max_time -0.5:
                #     print("TOOK TOO LONG")
                #     return min_score, best_move
                

                # calls method recursively with all potential moves after performing move
                score, _, = new_game.alpha_beta_minimax(
                    alpha, beta, Player.Attacker, current_depth + 1, branchingFactor, startTime)
                # replaces score if better one found
                if score < min_score:
                    min_score = score
                    best_move = move
                # including alpha beta pruning
                if(alpha is not None and beta is not None):
                    beta = min(beta, min_score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
            # add number of children to branching factor list to calculate avg branching factor later
            branchingFactor.append(numOfChildrens)
            return min_score, best_move

    # chooses heuristic based on condition entered 
    def chosen_heuristic(self):
        if self.options.heuristic == 0:
            return self.heuristicE0()
        elif self.options.heuristic == 1:
            return self.heuristicE1()
        elif self.options.heuristic == 2:
            return self.heuristicE2()
        else:
            return self.heuristicE0()
    
    # implementing heuristic E0 given in the handout
    def heuristicE0(self):
        return ((3 * (self.numOfVirusesAttacker - self.numOfTechsDefender)
          + 3 * (self.numOfFirewallAttacker - self.numOfFirewallDefender)
          + 3 * (self.numOfProgramsAttacker - self.numOfProgramsDefender)
          + 9999 * (self.numOfAIAttacker - self.numOfAIDefender)))

    # implementing heuristic E1 which is based on the health of a unit, its weight, and its position on the board
    # weight is added if units are in central positions on the board
    def heuristicE1(self):
        attackerScore=0
        defenderScore=0

        # adding weight to the score if positions are towards the center of the board
        centralWeight = 25

        # central tiles
        center_tiles = ['C2', 'D2', 'C3', 'B2', 'C1']

        # score for Attacker units, based on weight, health, and center control
        for (coord, unit) in self.player_units(Player.Attacker):
            if (unit.type == UnitType.Virus):
                attackerScore += 10
                attackerScore += unit.health * 7
            elif (unit.type == UnitType.Firewall):
                attackerScore += 2
                attackerScore += unit.health * 1
            elif (unit.type == UnitType.Program):
                attackerScore += 5
                attackerScore += unit.health * 2
            elif (unit.type == UnitType.AI):
                attackerScore += 9999
                attackerScore += unit.health * 20

            # adds to score if player is controlling center
            if coord.to_string() in center_tiles:
                attackerScore += centralWeight

        # score for Defender Units, based on weight, health, and center control
        for (coord, unit) in self.player_units(Player.Defender):
            if (unit.type == UnitType.Tech):
                defenderScore += 9
                defenderScore += unit.health * 6
            elif (unit.type == UnitType.Firewall):
                defenderScore += 2
                defenderScore += unit.health * 1
            elif (unit.type == UnitType.Program):
                defenderScore += 5
                defenderScore += unit.health * 2
            elif (unit.type == UnitType.AI):
                defenderScore += 9999
                defenderScore += unit.health * 20

            # adds to score if player is controlling center
            if coord.to_string() in center_tiles:
                defenderScore += centralWeight
            
        return attackerScore - defenderScore

    # implementing Heuristic E2 based on weight of unit and its position on the board
    # score is augmented based on a unit's distance to the opponent's AI
    def heuristicE2(self):
        score = 0

        coordDefenderAI_row = 0
        coordDefenderAI_col = 0
        coordAttackerAI_row = 0
        coordAttackerAI_col = 0

        # Find defender AI coord
        for (coord, unit) in self.player_units(Player.Attacker):
            if unit.type == UnitType.AI:
                coordDefenderAI_row = coord.row
                coordDefenderAI_col = coord.col

        # Find attacker AI coord
        for (coord, unit) in self.player_units(Player.Defender):
            if unit.type == UnitType.AI:
                coordAttackerAI_row = coord.row
                coordAttackerAI_col = coord.col

        # calculate health and and find how far the unit is from the opponents AI
        for (coord, unit) in self.player_units(Player.Attacker):
            if unit.is_alive():
                if (unit.type == UnitType.Virus):
                    score += unit.health * 20
                elif (unit.type == UnitType.Firewall):
                    score += unit.health * 5
                elif (unit.type == UnitType.Program):
                    score += unit.health * 10
                elif (unit.type == UnitType.AI):
                    score += unit.health * 999
                
                score += abs(coord.row - coordDefenderAI_row) + abs(coord.col - coordDefenderAI_col)
        
        # calculate health and and find how far the unit is from the opponents AI
        for (coord, unit) in self.player_units(Player.Defender):
            if unit.is_alive():
                if (unit.type == UnitType.Tech):
                    score -= unit.health * 19
                elif (unit.type == UnitType.Firewall):
                    score -= unit.health * 5
                elif (unit.type == UnitType.Program):
                    score -= unit.health * 10
                elif (unit.type == UnitType.AI):
                    score -= unit.health * 999

                score -= (abs(coord.row - coordAttackerAI_row) + abs(coord.col - coordAttackerAI_col))

        return score

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(
                    f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played+1:
                        move = CoordPair(
                            Coord(data['from']['row'], data['from']['col']),
                            Coord(data['to']['row'], data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(
                    f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

    def trace_each_action(self, src, dest):
        with open(self.gameTrace_path, 'a') as f:
            f.write("____________________________________________ \n \n")

            # human
            if (self.options.game_type == GameType.AttackerVsDefender) or (self.options.game_type == GameType.AttackerVsComp and self.next_player != Player.Defender) or (self.options.game_type == GameType.CompVsDefender and self.next_player != Player.Attacker):
                    f.write(f"Turn number: {self.turns_played + 1}/{self.options.max_turns} \n" +
                        f"Player: {self.next_player.name} \n" +
                        f"Action: {src} to {dest} \n \n" +
                        "New configuration of the board: \n" +
                        self.draw_board() + "\n"
                        )

            # computer
            if(self.options.game_type == GameType.CompVsComp) or (self.options.game_type == GameType.AttackerVsComp and self.next_player != Player.Attacker) or (self.options.game_type == GameType.CompVsDefender and self.next_player != Player.Defender):
                evaluations_by_depth = "Cumulative evals by depth: "
                evalutations_by_depth_percentage = "Cumulative % evals by depth: "
                for k in sorted(self.stats.evaluations_per_depth.keys()):
                    evaluations_by_depth += f"{k}:{self.stats.evaluations_per_depth[k]} "
                    evalutations_by_depth_percentage += f"{k}-{self.stats.evaluations_per_depth[k]/sum(self.stats.evaluations_per_depth.values())*100:0.1f}% "
                f.write(f"Turn number: {self.turns_played + 1}/{self.options.max_turns} \n" +
                        f"Player: {self.next_player.name} \n" +
                        f"Action: {src} to {dest} \n" +
                        f"AI time for action: {self.stats.elapsed_time :0.1f}s \n" +
                        f"AI heuristic score: {self.heuristic_score}\n \n" +
                        f"Cumulative evals: {sum(self.stats.evaluations_per_depth.values())} \n" +
                        evaluations_by_depth + "\n" +
                        evalutations_by_depth_percentage + "\n" +
                        f"Branching factor: {self.stats.avg_branching_factor} \n \n" +
                        "New configuration of the board: \n" +
                        self.draw_board() + "\n"
                        )



##############################################################################################################
# MAIN PROGRAM

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--max_turns', type=int,
                        help='maximum number of turns')
    # only minimax if alpha_beta is turned off
    parser.add_argument(
        '--alpha_beta_off', help='alpha beta pruning turned off [default: on]', action='store_false')
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--game_type', type=str, default="manual",
                        help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    parser.add_argument('--heuristic', type=int,
                        help='heuristic function options: 0/1/2')
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.max_turns is not None:
        options.max_turns = args.max_turns
    if args.alpha_beta_off is not None:
        options.alpha_beta = args.alpha_beta_off
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.broker is not None:
        options.broker = args.broker
    if args.heuristic is not None:
        options.heuristic = args.heuristic

    # create a new game
    game = Game(
        options=options,
        gameTrace_path=f'./gameTrace-{options.alpha_beta}-{options.max_time}-{options.max_turns}.txt'
    )

    if(options.game_type == GameType.AttackerVsComp) or (options.game_type == GameType.CompVsDefender) or (options.game_type == GameType.CompVsComp):
        for i in range(1, game.options.max_depth + 1):
            game.stats.evaluations_per_depth[i] = 0
       

    # game trace path
    try:
        # path = './gameTrace/gameTrace-{}-{}-{}.txt'.format(options.alpha_beta, options.max_time, options.max_turns)
        # If file does not exist, then create it
        if not os.path.exists(game.gameTrace_path):
            with open(game.gameTrace_path, 'w') as file:
                file.write('GAME TRACE \n \n')
                file.close()

        # If file exists, then clear its contents
        else:
            with open(game.gameTrace_path, 'w') as file:
                file.seek(0)
                file.truncate()
                file.write('GAME TRACE \n \n')
                file.close()

        # Write game parameters and initial configuration of the board
        with open(game.gameTrace_path, 'a') as file:
            file.write(
                'Game parameters: \n' +
                f'\tTimeout time (s): {options.max_time}\n' +
                f'\tMax number of turns: {options.max_turns}\n' +
                f'\tAlpha-beta (T/F): {options.alpha_beta}\n' +
                f'\tPlay modes: {options.game_type.name}\n' +
                f'\tHeuristic: {options.heuristic}' +
                '\n \n' +
                'INITIAL CONFIGURATION OF THE BOARD \n' +
                game.to_string() + '\n'
            )

    except FileNotFoundError:
        print("The 'gameTrace' directory does not exist")

    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        if winner is not None:
            with open(game.gameTrace_path, 'a') as file:
                file.write(
                    '\n \nGAME OVER!!!!!!! \n' +
                    f"{winner.name} wins in {game.turns_played} turns"
                )

            print(f"{winner.name} wins!")
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            elif move is None:
                if player == Player.Attacker:
                    print("Defender has won")
                else:
                    print("Attacker has won")
                break
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)

##############################################################################################################


if __name__ == '__main__':
    main()
