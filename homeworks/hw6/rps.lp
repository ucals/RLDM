\* Rock, Paper, Scissor *\
Maximize
OBJ: TotalValue
Subject To
_C1: PaperProbability + RockProbability + ScissorProbability = 1
_C2: - 3.79 PaperProbability + ScissorProbability - TotalValue >= 0
_C3: 3.79 RockProbability - 4.98 ScissorProbability - TotalValue >= 0
_C4: 4.98 PaperProbability - RockProbability - TotalValue >= 0
Bounds
PaperProbability <= 1
RockProbability <= 1
ScissorProbability <= 1
TotalValue free
End
