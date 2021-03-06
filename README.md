# Non-Shot-xG
Using a StatsBomb event based expected goals model with Metrica's tracking data to attribute non-shot-xG to every possession of each team.

[Non-Shot xG](https://thelastmananalytics.home.blog/2021/01/03/28-quantifying-non-shot-chances/)

## Why do they matter?
Goals, shots and expected goals are only counted if shots are actually taken.

## Problem
The problem is that there are some chances where a goal is likely but a player doesn’t shoot for some reason, they choose to pass, or hesitate etc.

How do we measure or quantify these Non-Shot chances?

![](home_goal_2_example.gif)

When a shot is taken, that is quantified and tracked in traditional metrics.

![](home_attack_104460.gif)

When there is no shot in a possession, that is lost. Non-shot expected goals takes the highest expected goal value assuming that the attacking palyer could have taken a shot.

This is NOT saying that the attacking player should always shoot rather than pass to get a better shot, but rather a way to quantify and track post-match.

### Credit
Karun Singh's Expected Threat concept: [Expected Threat](https://karun.in/blog/expected-threat.html)

Friends of Tracking and Laurie Shaw's Metrica Tracking Tutorials: [Friends of Tracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
