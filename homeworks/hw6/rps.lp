\* Rock, Paper, Scissor *\
Maximize
OBJ: TotalValue
Subject To
_C1: PaperProbability + RockProbability + ScissorProbability = 1
_C2: - 1.04 PaperProbability + 2.07 ScissorProbability - TotalValue >= 0
_C3: 1.04 RockProbability - 1.57 ScissorProbability - TotalValue >= 0
_C4: 1.57 PaperProbability - 2.07 RockProbability - TotalValue >= 0
Bounds
PaperProbability <= 1
RockProbability <= 1
ScissorProbability <= 1
TotalValue free
End
