clear;
two_user()
function []=two_user()
var = [0.05;0.15]; %noise
pmax = [1;1];
a = eye(2); %params in constraints
w = ones(2,1); %params in ojb fun

rho = 10e-4:1*10e-4:0.1;
len = length(rho);
R_tot = [];
warning_list = [];
for i = 1:1:len
    G = [1.2, rho(i);
        1.4*rho(i), 1];
    try
        p = outer_approxi_solve(G, w, a, var, pmax);
        G1=G-diag(diag(G));
        gamma = diag(diag(G))*p./(var + G1*p);
        R_tot = [R_tot, log(prod(gamma))];
    catch
        warning_list = [warning_list, i];
        warning("con2vert error happens...")
    end 
end
rho(warning_list) = [];
rho_log = log(rho);
xlabel = [1,10,30,85];
plot(rho_log, R_tot)
set(gca,  'Xtick', rho_log(xlabel))
set(gca, 'XtickLabel', rho(xlabel));
end


function []=three_user()
var = [0.05;0.05;0.05]; %noise
pmax = [1;1;1];
a = eye(3); %params in constraints
w = ones(3,1); %params in ojb fun
G = rand(3,3);
p = outer_approxi_solve(G, w, a, var, pmax);
end



