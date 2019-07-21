\* uCE-Q *\
Maximize
OBJ: 12 Action_0_Probability_Put + 9 Action_1_Probability_North
 + 9 Action_2_Probability_East
Subject To
_C1: Action_0_Probability_Put + Action_1_Probability_North
 + Action_2_Probability_East + Action_3_Probability_South = 1
_C2: - Action_0_Probability_Put + 2 Action_1_Probability_North >= 0
_C3: Action_2_Probability_East - 2 Action_3_Probability_South >= 0
_C4: - Action_0_Probability_Put + 2 Action_2_Probability_East >= 0
_C5: Action_1_Probability_North - 2 Action_3_Probability_South >= 0
Bounds
Action_0_Probability_Put <= 1
Action_1_Probability_North <= 1
Action_2_Probability_East <= 1
Action_3_Probability_South <= 1
End
