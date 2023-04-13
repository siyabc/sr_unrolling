function [power] = jsac2011_SINR_solve(G, w, a, var, pmax)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem parameter initialization
% L: number of users.
% G: channel gain.
% F: nonnegative matrix. F_{lj} = G_{lj}/G_{ll} if l \ne j, and F_{lj} = 0 if l = j
% var: noise power.
% v: nonnegative vector. v_l = var_l/G_{ll}
% pmax: upper bound of the weighted power constraints.
% a: weight of the power constraints.
% w: a probability vector that is a weight assigned to links to reflect priority.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[L,~] = size(G);
F = zeros(L,L);
v = zeros(L,1); 

% var = ones(L,1); %noise
% var = [0.05;0.15];

% pmax = pmax; %pmax
% a = eye(L); %param in constraints
% w = rand(L,1);
% w = w./sum(w);
% w = ones(L,1); %param in ojb fun

% for l = 1:1:L
%     for j = 1:1:L
%         if l~=j
%             G(l,j)=0.1+0.1*rand(1,1);
%         else
%             G(l,j)=0.6+0.3*rand(1,1);
%         end
%     end
% end

for l=1:1:L
    for j=1:1:L
        if l ~= j
            F(l,j)=G(l,j)/G(l,l);
        else
            F(l,j)=0;
        end
    end
    v(l) = var(l)/G(l,l);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% max_iteration: maximal number of iterations.
% epsilon: stopping criterion if ||p(k+1)-p(k)|| <= epsilon.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

max_iteration = 100;
epsilon = 0.01;

% Initialize the polyhedral convex set.
% A*gamma_tilde <= b is the polyhedral convex set.
A = []; b = [];
for l = 1:1:L
    B(:,:,l) = F+(1/pmax(l)).*(v*a(:,l)');
    
    [vec lamb] = eig(B(:,:,l));
    [val index]=max(max(lamb));
    x = vec(:,index);
    [vec lamb] = eig(B(:,:,l).');
    [val index]=max(max(lamb));
    y = vec(:,index);
    xy = (x.*y./sum(x.*y));
    A = [A;xy'];
    b = [b;-log(max(eig(B(:,:,l))))];
end
A = [A; -eye(L)]; b=[b; 100*ones(L,1)];

gamma_tilde = rand(L,1);
tolerance = 1;
k = 0;
power_evolution = [];

while tolerance >= epsilon || k <= max_iteration
    
    gamma_tilde_k = gamma_tilde;
    
    % Step 1 in Algorithm 1: compute the vertices.
    [V,nr] = con2vert(A,b);
    
    % Step 2 in Algorithm 1: select max{sumrate} from the vertices.
    sumrate = -Inf;
    s = size(V,1);
    for t = 1:1:s
        temp = sum(w.*log(ones(L,1)+exp(V(t,:)')));
        if temp > sumrate
            sumrate = temp;
            gamma_tilde = V(t,:)';
        end
    end
    
    % Step 3 in Algorithm 1: compute power.
    power = inv( eye(L) - diag(exp(gamma_tilde)) * F ) * diag(exp(gamma_tilde)) * v;
    power_evolution = [power_evolution ; power'];
    
    % Step 4 in Algorithm 1: let J = max{ diag(exp(gamma_tilde))*B(:,:,l) }.
    J=1;
    rho = 0;
    for l = 1:1:L
        Btemp = diag(exp(gamma_tilde))*B(:,:,l);
        if max(eig(Btemp)) > rho
            rho = max(eig(Btemp));
            J = l;
        end
    end
    
    % Step 5 in Algorithm 1: compute the Perron right and left eigenvectors
    % of diag(exp(gamma_tilde))*B(:,:,J).
    Btemp = diag(exp(gamma_tilde))*B(:,:,J);
    [vec lamb] = eig(Btemp);
    [val index]=max(max(lamb));
    x = vec(:,index);
    [vec lamb] = eig(Btemp.');
    [val index]=max(max(lamb));
    y = vec(:,index);
    xy = x.*y./sum(x.*y);
    % Step 6 in Algorithm 1: add a hyperplane to the polyhedral convex set.
    A = [A;xy'];
    b = [b;xy'*gamma_tilde-log(max(eig(Btemp)))];
    
    % Step 7 in Algorithm 1: update the iteration number.
    tolerance = norm(gamma_tilde_k-gamma_tilde);
    k = k+1;
end

% plot the power evolution.
% set(gca, 'Fontname', 'Times newman', 'Fontsize', 15);
% plot(1:1:k,power_evolution(:,1),'-',1:1:k,power_evolution(:,2),'-','linewidth',1.5);
% legend('User 1','User 2');
% xlim([0 k]);
% ylim([min(power)-2 max(power)+4]);
end