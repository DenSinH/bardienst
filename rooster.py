import itertools as it
import logging
import random
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Iterable

import yaml
from ortools.sat.python import cp_model
from pydantic import BaseModel, Field, PrivateAttr


def make_pair(v1, v2) -> frozenset:
    pair = frozenset((v1, v2))
    assert len(pair) == 2
    return pair


class Lid(BaseModel):
    favorieten: set[str] = Field(default_factory=set)
    overig: set[str] = Field(default_factory=set)

    def candidates(self) -> Iterable[str]:
        yield from (self.favorieten | self.overig)


Clique = Annotated[list[set[str]], Field(default_factory=list)]


class Cliques(BaseModel):
    favorieten: Clique
    overig: Clique


class Voorkeuren(BaseModel):
    cliques: Cliques
    leden: dict[str, Lid | None] = Field(default_factory=dict)
    _vars: dict[frozenset[str], cp_model.IntVar] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        logging.info("Voorkeuren repareren...")
        self._missing_members()
        self._cliques()
        self._make_symmetric()
        self._promote_favorites()
        logging.info(f"{len(self.leden)} leden")

    def _missing_members(self):
        """Create empty members for any missing members"""
        leden = set(self.leden)

        # repair 'empty' members
        for name in leden:
            if self.leden[name] is None:
                self.leden[name] = Lid()

        for name, lid in self.leden.items():
            leden |= lid.favorieten | lid.overig
        for clique in self.cliques.favorieten:
            leden |= clique
        for clique in self.cliques.overig:
            leden |= clique

        for name in leden:
            if name not in self.leden:
                logging.info(f"Voeg missend lid '{name}' toe")
                self.leden[name] = Lid()

    def _cliques(self):
        """Add people from cliques to each others preferences"""
        for clique in self.cliques.favorieten:
            for lid, other in it.permutations(clique, 2):
                self.leden[lid].favorieten.add(other)
                self.leden[other].favorieten.add(lid)

        for clique in self.cliques.overig:
            for lid, other in it.permutations(clique, 2):
                self.leden[lid].overig.add(other)
                self.leden[other].overig.add(lid)

    def _make_symmetric(self):
        """Fix leden mapping by making the favorieten and overig lists symmetric."""
        for name, lid in list(self.leden.items()):
            for other in lid.favorieten:
                if name not in self.leden[other].favorieten:
                    logging.info(f"Voeg '{name}' toe aan '{other}' favorieten")
                self.leden[other].favorieten.add(name)
            for other in lid.overig:
                if name not in self.leden[other].overig:
                    logging.info(f"Voeg '{name}' toe aan '{other}' overig")
                self.leden[other].overig.add(name)

    def _promote_favorites(self):
        """Remove any members from the 'overig' mapping if they are already 'favorites' """
        for name, lid in self.leden.items():
            lid.overig -= lid.favorieten
            lid.overig.discard(name)
            lid.favorieten.discard(name)

    def _init_vars(self, model: cp_model.CpModel):
        """Initialize cp model variables for members"""
        for name, lid in self.leden.items():
            for other in lid.candidates():
                pair = make_pair(name, other)
                if pair not in self._vars:
                    self._vars[pair] = model.new_bool_var(str(sorted(pair)))

    def construct_problem(self, model: cp_model.CpModel):
        """Construct the problem given the preferences, then return the objective"""
        logging.info("Model aanmaken")
        self._init_vars(model)

        # members can be scheduled at most once
        for name, lid in self.leden.items():
            model.add_at_most_one(
                *(self._vars[make_pair(name, other)] for other in lid.candidates())
            )

        is_scheduled: dict[str, cp_model.IntVar] = {}
        for name, lid in self.leden.items():
            is_scheduled[name] = model.new_bool_var(f"scheduled: {name}")
            model.add_max_equality(
                is_scheduled[name],
                (self._vars[make_pair(name, other)] for other in lid.candidates())
            )

        logging.info(
            f"Model heeft {len(model.proto.variables)} variabelen "
            f"en {len(model.proto.constraints)} vergelijkingen"
        )

        # HUGE penalty for not scheduling members
        objective = 100000 * sum(is_scheduled.values())
        for name, lid in self.leden.items():
            for other in lid.favorieten:
                # bunus if favorite members are together
                objective += self._vars[make_pair(name, other)]

        model.maximize(objective)

    def output_schedule(self, solver: cp_model.CpSolver, shuffled: bool = True):
        """Output a single schedule from the solver"""

        # gather formed pairs
        pairs = [
            pair for pair, var in self._vars.items()
            if solver.value(var)
        ]
        if shuffled:
            random.shuffle(pairs)

        # output schedule
        for pair in pairs:
            print(", ".join(pair))

        # output unscheduled members
        for name, lid in self.leden.items():
            is_scheduled = any(
                solver.value(self._vars[frozenset((name, other))])
                for other in lid.candidates()
            )
            if not is_scheduled:
                print(name, "is niet ingeroosterd")


def load_preferences(filename: PathLike | str) -> Voorkeuren:
    """Load preferences from a file"""
    logging.info(f"Voorkeuren laden van {Path(filename).absolute()}")
    with Path(filename).open("r") as f:
        return Voorkeuren(**yaml.safe_load(f))


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s %(levelname)-8s] %(message)s",
    )
    voorkeuren = load_preferences("voorkeuren.yml")
    model = cp_model.CpModel()
    voorkeuren.construct_problem(model)


    class SolutionLogger(cp_model.CpSolverSolutionCallback):

        def on_solution_callback(self):
            logging.info(
                f"Oplossing objective value {self.objective_value} "
                f"en best objective bound {self.best_objective_bound}"
            )


    solver = cp_model.CpSolver()
    solver.best_bound_callback = lambda bound: logging.info(f"Best objective bound {bound}")
    solver.parameters.max_time_in_seconds = 20
    solver.parameters.permute_variable_randomly = True
    solver.parameters.randomize_search = True
    solver.parameters.permute_variable_randomly = True
    status = solver.solve(model, SolutionLogger())

    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise Exception("Kan geen rooster maken met deze instellingen...")

    voorkeuren.output_schedule(solver)
