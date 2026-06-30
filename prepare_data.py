import pandas as pd

# ── Load & filter IPL only ─────────────────────────────────────
df = pd.read_csv("IPL.csv", low_memory=False)
df = df[df["event_name"] == "Indian Premier League"]
df = df[df["year"] <= 2025]


# ── BATTING STATS ──────────────────────────────────────────────

# Innings-level first (to get Highest_Score and dismissals)
batting_innings = (
    df.groupby(["batter", "year", "match_id", "innings", "batting_team"])
    .agg(
        innings_runs=("runs_batter", "sum"),
        balls=("valid_ball", "sum"),
        fours=("runs_batter", lambda x: (x == 4).sum()),
        sixes=("runs_batter", lambda x: (x == 6).sum()),
        dismissed=("player_out", lambda x: x.notna().any()),
    )
    .reset_index()
)

# Player-Year batting aggregate
batting = (
    batting_innings.groupby(["batter", "year", "batting_team"])
    .agg(
        Matches=("match_id", "nunique"),
        Runs=("innings_runs", "sum"),
        Balls=("balls", "sum"),
        Fours=("fours", "sum"),
        Sixes=("sixes", "sum"),
        Highest_Score=("innings_runs", "max"),
        dismissals=("dismissed", "sum"),
    )
    .reset_index()
)

# Average = Runs / dismissals (avoid division by zero)
batting["Average"] = (batting["Runs"] / batting["dismissals"].replace(0, 1)).round(2)
batting["Strike_Rate"] = (
    (batting["Runs"] / batting["Balls"].replace(0, 1)) * 100
).round(2)

# Team = most frequent team that year for that player
team_per_player = (
    batting_innings.groupby(["batter", "year"])["batting_team"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
    .rename(columns={"batting_team": "Team"})
)

batting = batting.drop(columns=["batting_team", "dismissals"])
batting = batting.merge(team_per_player, on=["batter", "year"])


# ── BOWLING STATS ──────────────────────────────────────────────

# Only valid bowler wickets
valid_wickets = [
    "caught",
    "bowled",
    "lbw",
    "stumped",
    "caught and bowled",
    "hit wicket",
]

bowling_innings = (
    df.groupby(["bowler", "year", "match_id", "innings"])
    .agg(
        balls_bowled=("valid_ball", "sum"),
        runs_conceded=("runs_bowler", "sum"),
        wickets=("wicket_kind", lambda x: x.isin(valid_wickets).sum()),
    )
    .reset_index()
)

bowling = (
    bowling_innings.groupby(["bowler", "year"])
    .agg(
        balls_bowled=("balls_bowled", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        Wickets=("wickets", "sum"),
    )
    .reset_index()
)

bowling["Overs"] = (bowling["balls_bowled"] // 6) + (bowling["balls_bowled"] % 6) * 0.1
bowling["Economy"] = (bowling["runs_conceded"] / bowling["Overs"].replace(0, 1)).round(
    2
)
bowling = bowling.drop(columns=["balls_bowled", "runs_conceded"])


# ── MERGE batting + bowling ────────────────────────────────────

final = batting.merge(
    bowling, left_on=["batter", "year"], right_on=["bowler", "year"], how="outer"
)

# Unify Player name column (batter or bowler depending on which side matched)
final["Player"] = final["batter"].combine_first(final["bowler"])
final["Year"] = final["year"]

# Fill missing stats with 0 (pure batters have no bowling stats, vice versa)
final[
    ["Runs", "Balls", "Fours", "Sixes", "Highest_Score", "Average", "Strike_Rate"]
] = final[
    ["Runs", "Balls", "Fours", "Sixes", "Highest_Score", "Average", "Strike_Rate"]
].fillna(
    0
)

final[["Wickets", "Overs", "Economy"]] = final[["Wickets", "Overs", "Economy"]].fillna(
    0
)

final["Matches"] = final["Matches"].fillna(0)
final["Team"] = final["Team"].fillna("Unknown")

# Keep only required columns
final = final[
    [
        "Player",
        "Year",
        "Team",
        "Matches",
        "Runs",
        "Balls",
        "Average",
        "Strike_Rate",
        "Fours",
        "Sixes",
        "Highest_Score",
        "Wickets",
        "Overs",
        "Economy",
    ]
]


# ── FANTASY POINTS ─────────────────────────────────────────────

# ── Batting milestones (innings-level) ──
batting_innings['milestone'] = batting_innings['innings_runs'].apply(
    lambda x: 32 if x >= 100 else (16 if x >= 50 else (8 if x >= 25 else 0))
)
batting_innings['sr_bonus'] = batting_innings.apply(
    lambda r: 6 if r['balls'] >= 10 and r['innings_runs']/r['balls']*100 > 140
    else (4 if r['balls'] >= 10 and r['innings_runs']/r['balls']*100 >= 120
    else (-2 if r['balls'] >= 10 and r['innings_runs']/r['balls']*100 <= 50
    else (-6 if r['balls'] >= 10 and r['innings_runs']/r['balls']*100 < 30 else 0))), axis=1
)

batting_bonus = (
    batting_innings.groupby(['batter', 'year'])
    .agg(Milestone_Points=('milestone', 'sum'), SR_Bonus=('sr_bonus', 'sum'))
    .reset_index().rename(columns={'batter': 'Player', 'year': 'Year'})
)

# ── Bowling milestones (innings-level) ──
lbw_bowled = df[df['wicket_kind'].isin(['lbw', 'bowled'])].copy()
lbw_bowled_bonus = (
    lbw_bowled.groupby(['bowler', 'year'])
    .size().reset_index(name='LBW_Bowled_Count')
    .rename(columns={'bowler': 'Player', 'year': 'Year'})
)

bowling_innings['wicket_haul_bonus'] = bowling_innings['wickets'].apply(
    lambda x: 16 if x >= 4 else (8 if x >= 3 else 0)
)
bowling_innings['economy_bonus'] = bowling_innings.apply(
    lambda r: (6 if r['balls_bowled'] >= 30 and r['runs_conceded']/(r['balls_bowled']/6) < 3.5
    else (-3 if r['balls_bowled'] >= 30 and r['runs_conceded']/(r['balls_bowled']/6) > 11 else 0)), axis=1
)

bowling_bonus = (
    bowling_innings.groupby(['bowler', 'year'])
    .agg(Haul_Bonus=('wicket_haul_bonus', 'sum'), Economy_Bonus=('economy_bonus', 'sum'))
    .reset_index().rename(columns={'bowler': 'Player', 'year': 'Year'})
)

# ── Merge bonuses into final ──
final = final.merge(batting_bonus, on=['Player', 'Year'], how='left')
final = final.merge(lbw_bowled_bonus, on=['Player', 'Year'], how='left')
final = final.merge(bowling_bonus, on=['Player', 'Year'], how='left')
final[['Milestone_Points','SR_Bonus','LBW_Bowled_Count','Haul_Bonus','Economy_Bonus']] = \
    final[['Milestone_Points','SR_Bonus','LBW_Bowled_Count','Haul_Bonus','Economy_Bonus']].fillna(0)

# ── Calculate Fantasy Points ──
final['Batting_Fantasy_Points'] = (
    final['Runs'] * 1 +
    final['Fours'] * 4 +
    final['Sixes'] * 6 +
    final['Milestone_Points'] +
    final['SR_Bonus']
)

final['Bowling_Fantasy_Points'] = (
    final['Wickets'] * 25 +
    final['LBW_Bowled_Count'] * 8 +
    final['Haul_Bonus'] +
    final['Economy_Bonus']
)

final['Allrounder_Fantasy_Points'] = (
    final['Batting_Fantasy_Points'] + final['Bowling_Fantasy_Points']
)

final['Fantasy_Points'] = final['Allrounder_Fantasy_Points']

# ── Save ───────────────────────────────────────────────────────
final = final.sort_values("Year").reset_index(drop=True)
final.to_csv("ipl_prepared.csv", index=False)
print("Done! Shape:", final.shape)
print(final.head())