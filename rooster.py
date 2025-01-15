import logging
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Annotated, Any

import yaml
from ortools.sat.python import cp_model
from pydantic import BaseModel, Field, ConfigDict

LEDEN_PER_WEEK = 2


class Lid(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    favorieten: set[str] = Field(default_factory=set)
    overig: set[str] = Field(default_factory=set)
    vars_: list[cp_model.IntVar] = Field(default_factory=list)


class Voorkeuren(BaseModel):
    weken: int = Field(default=0)
    leden: dict[str, Lid] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        logging.info("Voorkeuren repareren...")
        self._missing_members()
        self._make_symmetric()
        self._promote_favorites()
        self.weken = len(self.leden) // LEDEN_PER_WEEK
        logging.info(f"{len(self.leden)} leden, rooster voor {self.weken} weken")

    def _missing_members(self):
        """Create empty members for any missing members"""
        for lid in list(self.leden.values()):
            for other in lid.favorieten | lid.overig:
                if other not in self.leden:
                    logging.info(f"Voeg missend lid '{other}' toe")
                    self.leden[other] = Lid()

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
            for week in range(self.weken):
                lid.vars_.append(model.new_bool_var(str((name, week))))

    def construct_problem(self, model: cp_model.CpModel) -> cp_model.LinearExpr:
        """Construct the problem given the preferences, then return the objective"""
        logging.info("Model aanmaken")
        self._init_vars(model)

        for week in range(self.weken):
            # 2 members per week have to be scheduled
            model.add(
                sum(lid.vars_[week] for lid in self.leden.values()) == LEDEN_PER_WEEK
            )

            # only schedule together members with other allowed members
            for name, lid in self.leden.items():
                # member_scheduled => one of the compatible members scheduled the
                # same week
                model.add_bool_or(
                    *(self.leden[other].vars_[week] for other in lid.favorieten | lid.overig)
                ).only_enforce_if(lid.vars_[week])

        # members can be scheduled at most once
        for lid in self.leden.values():
            model.add_at_most_one(lid.vars_)
        logging.info(
            f"Model heeft {len(model.proto.variables)} variabelen "
            f"en {len(model.proto.constraints)} vergelijkingen"
        )

        objective = 0
        for week in range(self.weken):
            for name, lid in self.leden.items():
                for other in lid.favorieten:
                    # are (favorite) members together this week?
                    together = model.new_bool_var(str((name, other, week)))

                    # together <=> lid[week] && other[week]
                    # we then maximize the amount of favorites that work together
                    model.add_bool_or(
                        lid.vars_[week].Not(), self.leden[other].vars_[week].Not()
                    ).only_enforce_if(together.Not())
                    model.add_bool_and(
                        lid.vars_[week], self.leden[other].vars_[week]
                    ).only_enforce_if(together)

                    objective += together
        return objective

    def output_schedule(self, solver: cp_model.CpSolver):
        """Output a single schedule from the solver"""
        for week in range(self.weken):
            leden = tuple(
                name for name, lid in self.leden.items()
                if solver.value(lid.vars_[week])
            )
            print(week, *leden)

        for name, lid in self.leden.items():
            if not any(solver.value(lid.vars_[week]) for week in range(self.weken)):
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
    voorkeuren = load_preferences("voorkeuren.example.yml")
    model = cp_model.CpModel()
    objective = voorkeuren.construct_problem(model)

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise Exception("Kan geen rooster maken met deze instellingen...")

    voorkeuren.output_schedule(solver)
