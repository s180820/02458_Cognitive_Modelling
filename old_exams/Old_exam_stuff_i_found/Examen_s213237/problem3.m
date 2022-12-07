%% problem 3

% P(R|s)=P(s|R)*P(R)
P_R=0.75;
% we have a normal distribution centered at 5 cm, with sd equals 10
% Right Hole is at 15 cm from that, so we compute the likelihood
Ps_R=normpdf(15,0,10);
PR_s=Ps_R*P_R

% Same procedure with Left hole
P_L=0.25;
% now the hole is at 5 cm from where the normal dist is located
Ps_L=normpdf(-5,0,10);
PL_s=Ps_L*P_L
% We could also compute the likelihoods by centering the distribution at 5
% normpdf(0,5,10) = normpdf(-5,0,10)

% For Maximum Posteriori Probability
PR_s>PL_s

% So mouse is gonna target the Right Hole

%% Combine auditory with visual system








