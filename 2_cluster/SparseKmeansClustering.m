function [IDX, w] = SparseKmeansClustering(X, K, s)
% SparseKmeansClustering - performing sparse k-means clustering using
% algorithm in (Witten, et al. JASA 2010)
% Input:
% X - data matrix (sample x feature)
% K - number of clusters
% s - bound on the feature weight coefficients (>1)
% Output:
% IDX - cluster index for each sample
% W - feature weight coefficients
%
% Wei Wu, Stanford University, April 15, 2019
% Email: wwumed@stanford.edu

[N, p] = size(X);

%% Initialize w
w = 1/sqrt(p)*ones(p, 1);
w0 = w;
fprintf('Starting sparse K-means clustering ... \n');
kk = 1;
%% iteratively optimze IDX and w
while true
    
    fprintf('Iteration %d \n', kk);
    % optimize IDX by k-means
    Xw = repmat(sqrt(w)', N, 1).*X;
    IDX = kmeans(Xw, K, 'replicate', 100);
    
    % optimize w
    d = zeros(N, N, p);
    for i = 1:N
        for ip = 1:N
            d(i, ip, :) = (X(i, :) - X(ip, :)).^2; 
        end
    end
    
    a = zeros(p, 1);
    aplus = zeros(p, 1);
    delta = 0;
    for j = 1:p     
        wcd = zeros(K, 1);
        for k = 1:K
            Nk = sum(IDX == k);
            wcd(k) = 1/Nk*sum(sum(d(IDX == k, IDX == k, j)));
        end
        a(j) = 1/N*sum(sum(d(:, :, j))) - sum(wcd);
        aplus(j) = max([a(j), 0]);
        w(j) = max(abs(aplus(j)) - delta, 0);
    end
    w = w/norm(w);
   
    if norm(w, 1) > s
        % find delta so norm(w, 1) = s
        f  = @(x) solvedelta(x, w, s);
        options = optimoptions('fsolve','Display','none');
        delta = fsolve(f, delta, options);      
        for j = 1:p
            w(j) = max(w(j) - delta, 0);
        end
        w = w/norm(w);
    end
    
    % check convergence
    fprintf('Relative change of w is %4.4f\n', norm(w - w0, 1)/norm(w0, 1));
    if norm(w - w0, 1)/norm(w0, 1) < 1e-4
        fprintf('Success! Algorithm is converged!\n');
        break;
    else
        w0 = w;
    end
    kk = kk + 1;
    
    if kk == 100
        fprintf('Return!\n');
        return
    end
end

function F = solvedelta(x, w, s)
f = zeros(1, length(w));
for j = 1:length(w)
    f(j) = max(w(j) - x, 0);
end
f = f/norm(f);
F = double(norm(f, 1) - s); 
