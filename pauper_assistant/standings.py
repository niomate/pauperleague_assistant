from abc import abstractmethod
import os
import json
import pandas as pd
import pathlib as pl

from pydantic import BaseModel

root = pl.Path(__file__).parents[1]


class TournamentResultPlayer(BaseModel):
    name: str
    points: int
    wins: int
    losses: int
    draws: int


class TournamentResult(BaseModel):
    ranks: list[TournamentResultPlayer]
    rounds: int


class RankingPlayer(BaseModel):
    name: str
    rounds_played: int
    average_points: float
    wins: int
    losses: int
    draws: int


class Ranking(BaseModel):
    ranks: list[RankingPlayer]


class RankingMethod:
    @abstractmethod
    def __call__(self, standings: list[TournamentResult]): ...

    def combine_standings(self, standings: list[TournamentResult]):
        combined_standings = pd.DataFrame()
        for fnm in standings:
            df = pd.DataFrame(fnm)
            combined_standings = pd.concat([combined_standings, df])
        return combined_standings


class HighestAveragePoints(RankingMethod):
    def __call__(self, standings: list[TournamentResult]) -> Ranking:
        combined_standings = self.combine_standings(standings)
        combined_standings = combined_standings.groupby("Name").sum()

        combined_standings["Rounds Played"] = (
            combined_standings["W"] + combined_standings["L"] + combined_standings["D"]
        )

        combined_standings["Average Points"] = (
            combined_standings["Points"] / combined_standings["Rounds Played"]
        )

        combined_standings = combined_standings.sort_values(by="Average Points", ascending=False)
        return combined_standings


class HighestAveragePointsTop10(RankingMethod):
    def __call__(self, standings: list[TournamentResult]) -> Ranking:
        combined_standings = self.combine_standings(standings)
        combined_standings = combined_standings.sort_values(by="Points", ascending=False).groupby("Name").head(10).groupby("Name").sum()

        combined_standings["Rounds Played"] = (
            combined_standings["W"] + combined_standings["L"] + combined_standings["D"]
        )

        combined_standings["Average Points"] = (
            combined_standings["Points"] / combined_standings["Rounds Played"]
        )

        combined_standings = combined_standings.sort_values(by="Average Points", ascending=False)
        return combined_standings


highest_average_points = HighestAveragePoints()
highest_average_points_top10 = HighestAveragePointsTop10()


def load() -> list[TournamentResult]:
    if os.path.exists(root / "resources" / "rankings.json"):
        with open(root / "resources" / "rankings.json", "r") as f:
            return json.load(f)
    return []


def store(standings: list[TournamentResult]):
    with open(root / "resources" / "rankings.json") as f:
        json.dump(standings, f)


def update(latest_rankings: pd.DataFrame):
    standings = load()
    standings.append(json.loads(latest_rankings.to_json(orient="records")))
    store(standings)

    return standings


def calculate_standings(
    standings: list[TournamentResult],
    ranking_method: RankingMethod = highest_average_points,
) -> Ranking:
    return ranking_method(standings)
