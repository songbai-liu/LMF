classdef LMF5 < PROBLEM
% <multi/many> <real> <large/none>
% Large-scale Multiobjective Optimization Problem
% type_fm --- 2 --- type for formulation model, 0 for addition, 1 for multiplication, and 2 for mixed model
% type_lk --- 2 --- type for variable linkage, 0 for linear linkage, 1 for nonlinear linkage, and 2 for mixed linkage
% type_dg --- 1 --- type for deep grouping distance-related variables, 0 for even grouping and 1 for nonuniform grouping
% type_cv --- 1 --- type for contribution of variables, 0 for balanced contribution and 1 for unbalanced contribution

%------------------------------- Reference --------------------------------
% Songbai Liu，et al. "Evolutionary Large-Scale Multiobjective Optimization: Benchmarks and Algorithms," IEEE TEVC, 2021.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    properties(Access = private)
        nip; % Number of unique variables in each position-related variable group
        nop; % Number of overlapping variables in each position-related variable group
        nsp; % Number of shared variables in each position-related variable group
        nid; % Number of independent variables in each distance-related variable group
        nod; % Number of overlapping variables in each distance-related variable group
        nsd; % Number of shared variables in each distance-related variable group
        K;%Number of position-related variables
        L;%Number of distance-related variables
		type_fm = 2;
        type_lk = 2;
        type_dg = 1;
        type_cv = 1;
    end
    methods
       %% Default settings of the problem
        function Setting(obj)
            % Parameter setting
			obj.type_fm = obj.ParameterSet(2);
            obj.type_lk = obj.ParameterSet(2);
            obj.type_dg = obj.ParameterSet(1);
            obj.type_cv = obj.ParameterSet(1);
            if isempty(obj.M)
                obj.M = 3;
            end
            if isempty(obj.D)
                obj.D = 100*obj.M;
            end
            obj.nip = 3*ones(1,obj.M-1);
            if obj.M == 2
                obj.nop = 0*ones(1,obj.M-1);
            else
                obj.nop = 1*ones(1,obj.M-1);
            end
            obj.nsp = 1;
            obj.K = sum(obj.nip) + obj.nsp;
            obj.L = obj.D - obj.K;
            obj.lower = [-1*ones(1,obj.K),zeros(1,obj.L)];
            obj.upper = [ones(1,obj.K),2*ones(1,obj.L)];
            obj.encoding = 'real';
            % Calculate the number of independent distance-related
            % variables in each group
            c = LogisticMap(3.8*0.1*(1-0.1),3.8,obj.M);
            nsd_least = 2;%make sure the number of nsd is at least 2
            proportion = 0.2;%define the number of nod as a percentage of the number of nid
            obj.nid = floor(c.*(obj.L-nsd_least));
            obj.nod = [floor(obj.nid(end).*proportion),floor(obj.nid(1:end-1).*proportion)];
            obj.nsd = obj.L - sum(obj.nid);
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            N = size(PopDec,1);
            D = size(PopDec,2);
            M = obj.M;
            %nonuniform grouping of the variables
            gp = Grouping(obj.nip,obj.nop,obj.nsp,M-1,1);
            gd = Grouping(obj.nid,obj.nod,obj.nsd,M,obj.K+1);
            % calculate the shape function H
            yp = zeros(N,M-1); %using the meta variable yp
            for i = 1 : M-1
                yp(:,i) = yp(:,i) + rt_sum(PopDec(:,1:obj.K),gp{i});
            end
            H = ones(N,M);
            for i = 1 : M
                for j = 1 : M - i
                    H(:,i) = H(:,i).*cos(0.5.*pi.*yp(:,j));
                end
                if i ~= 1
                    aux = M - i + 1;
                    H(:,i) = H(:,i).*sin(0.5.*pi.* yp(:,aux));
                end
                if i ~= M
                   H(:,i) = H(:,i).^4; 
                else
                   H(:,i) = H(:,i).^2; 
                end
            end
             %varialbe linkages between distance-ralated variables and
            if obj.type_lk == 0 % linear linkages
                for i = obj.K+1 : D
                    PopDec(:,i) = (1+(i-obj.K)./obj.L).*(PopDec(:,i)-obj.lower(i)) - yp(:,1).*(obj.upper(i)-obj.lower(i));   
                end 
            elseif obj.type_lk == 1 % nonlinear linkages
                for i = obj.K+1 : D
                    PopDec(:,i) = (1+cos((i-obj.K)./obj.L*pi/2)).*(PopDec(:,i)-obj.lower(i)) - yp(:,1).*(obj.upper(i)-obj.lower(i));
                end
            else                   %mixed linkages
                for i = obj.K+1 : 2 : D
                    PopDec(:,i) = (1+(i-obj.K)./obj.L).*(PopDec(:,i)-obj.lower(i)) - yp(:,1).*(obj.upper(i)-obj.lower(i));
                end
                for i = obj.K+2 : 2 : D
                    PopDec(:,i) = (1+cos((i-obj.K)./obj.L*pi/2)).*(PopDec(:,i)-obj.lower(i)) - yp(:,1).*(obj.upper(i)-obj.lower(i));
                end
            end
            x = PopDec;
            %Deep Grouping of the distance-related variables in each group
            yd = cell(1,M);
            a1 = 5;%Define the value of the first entry in the sequence
            d1 = 1;%Define the value of the variance of the sequence
			nk1 = 5; %Number of groups for uniformly deep grouping
            for i = 1 : 2 : M
                if obj.type_dg == 1 %nonuniformly deep grouping without knowing the number of groups
                    dg = DeepGrouping(gd{i},a1,d1);
                else %evenly deep grouping, and the number of groups is preknown, i.e., nk1
                    dg = UniformGrouping(gd{i},nk1);
                end
                yd{i} = zeros(N,length(dg));
                for j = 1 : 3 : length(dg)
                    yd{i}(:,j) = Sphere(x,dg{j});
                end
                for j = 2 : 3 : length(dg)
                    yd{i}(:,j) = Schwefel2(x,dg{j});
                end
                for j = 3 : 3 : length(dg)
                    yd{i}(:,j) = Schwefel221(x,dg{j});
                end
            end
            a2 = 5;%Define the value of the first entry in the sequence
            d2 = 2;%Define the value of the variance of the sequence
			nk2 = 5; %Number of groups for uniformly deep grouping
            for i = 2 : 2 : M
                if obj.type_dg == 1 %nonuniformly deep grouping without knowing the number of groups
                    dg = DeepGrouping(gd{i},a2,d2);
                else %evenly deep grouping, and the number of groups is preknown, i.e., nk2
                    dg = UniformGrouping(gd{i},nk2);
                end
                yd{i} = zeros(N,length(dg));
                for j = 1 : 3 : length(dg)
                    yd{i}(:,j) = Sphere(x,dg{j});
                end
                for j = 2 : 3 : length(dg)
                    yd{i}(:,j) = Ackley(x,dg{j});
                end
                for j = 3 : 3 : length(dg)
                    yd{i}(:,j) = Rastrigin(x,dg{j});
                end
            end
            % calculate the distance function G
            G = zeros(N,M);
            for i = 1 : 2 : M
                len1 = size(yd{i},2);
                w1 = LogisticMap(0.23,3.7,len1);
                if obj.type_cv == 1 %imbalanced contribution of variables to the landscap function
                    G(:,i) = G(:,i) + WeightedSum(yd{i},w1);
                else %balanced contribution of variables to the landscap function
                    G(:,i) = G(:,i) + EvenSum(yd{i});
                end
            end
            for i = 2 : 2 : M
                len2 = size(yd{i},2);
                w2 = LogisticMap(0.23,3.75,len2);
                if obj.type_cv == 1 %imbalanced contribution of variables to the landscap function
                    G(:,i) = G(:,i) + WeightedSum(yd{i},w2);
                else %balanced contribution of variables to the landscap function
                    G(:,i) = G(:,i) + EvenSum(yd{i});
                end
            end
            % evaluate the objective values
            if obj.type_fm == 0 %addition model
                PopObj = G + H;    
            elseif obj.type_fm == 1 %multiplication model
                PopObj = (1+G).*H;    
            else %mixed model
                for i = 1 : 2 : M
                    PopObj(:,i) = H(:,i).*(1 + G(:,i));
                end
                for i = 2 : 2 : M
                    PopObj(:,i) = H(:,i) + G(:,i);
                end    
            end
        end
       %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
            R    = UniformPoint(N,obj.M).^2;
            temp = sum(sqrt(R(:,1:end-1)),2) + R(:,end);
            R    = R./[repmat(temp.^2,1,size(R,2)-1),temp];
        end
        %% Generate the image of Pareto front
        function R = GetPF(obj)
            if obj.M == 2
                R = obj.GetOptimum(100);
            elseif obj.M == 3
                a = linspace(0,pi/2,10)';
                x = sin(a)*cos(a');
                y = sin(a)*sin(a');
                z = cos(a)*ones(size(a'));
                R = {x.^4,y.^4,z.^2};
            else
                R = [];
            end
        end
    end
end

function fc = LogisticMap(c1,r,num)
    fc = zeros(1,num);
    fc(1) = c1;
    for i = 1 : num-1
        fc(i+1) = r.*fc(i).*(1-fc(i));
    end
    sum_c = sum(fc);
    fc = fc./sum_c;
end

function f = rt_sum(x,g)
    l = length(g(:));
    [s1,~] = size(x);
    f = zeros(s1,1);
    for i = 1 : l
       f = f + x(:,g(i)); 
    end
    f = abs(f)./l;
end

function f = WeightedSum(x,w)
    [s1,s2] = size(x);
    f = zeros(s1,1);
    for i = 1 : s2
       f = f + x(:,i).*w(i); 
    end
end

function f = EvenSum(x)
    [s1,s2] = size(x);
    f = zeros(s1,1);
    for i = 1 : s2
       f = f + x(:,i).*(1./s2); 
    end
end

%Unimodal and Separable
function gx = Sphere(x,g)
    l = length(g(:));
    [s1,~] = size(x);
    gx = zeros(s1,1);
    for i = 1 : l
       gx = gx + x(:,g(i)).^2; 
    end
    gx = gx./l;
end

%Unimodal and Non-Separable
function gx = Schwefel221(x,g)
    l = length(g(:));
    gx = abs(x(:,g(1)));
    for i = 2 : l
        gx = max(gx,abs(x(:,g(i))));
    end
end

%Unimodal and Non-Separable
function gx = Schwefel2(x,g)
    l = length(g(:));
    [s1,~] = size(x);
    gx = zeros(s1,1);
    for i = 1 : l
        midx = zeros(s1,1);
        for j = 1 : i
            midx = midx + x(:,g(j));
        end
        gx = gx + midx.^2;
    end
end

%Multimodal and Separable
function gx = Ackley(x,g)
    l = length(g(:));
    [s1,~] = size(x);
    gx = zeros(s1,1);
    sum1 = zeros(s1,1);
    sum2 = zeros(s1,1);
    for i = 1 : l
       sum1 = sum1 + x(:,g(i)).^2./l;
       sum2 = sum2 + cos(2.*pi.*x(:,g(i)))/l;
    end
    gx = gx + 20 - 20.*exp(-0.2.*sqrt(sum1)) + exp(1) - exp(sum2);
    gx = gx./l;
end

%Multimodal and Separable
function gx = Rastrigin(x,g)
    l = length(g(:));
    [s1,~] = size(x);
    gx = zeros(s1,1);
    for i = 1 : l
       gx = gx + x(:,g(i)).^2 + 10 - 10.*cos(2.*pi.*x(:,g(i)));
    end
    gx = gx./l;
end

function g = Grouping(ni,no,ns,ng,point)
    pointer = point;
    len = zeros(1,ng);
    len = len + ni + no;
    len = len + ns;
    g = cell(1,ng);
    for i = 1 : ng
        for j = no(i) + ns + 1 : len(i)
            g{i}(j) = pointer;
            pointer = pointer+1; 
        end
    end
    for i = 1 : ng
        for j = 1 : no(i)
            if i == 1
                g{i}(j) = g{ng}(len(ng)-j+1);
            else
                g{i}(j) = g{i-1}(len(i-1)-j+1);
            end 
        end
    end
    for j = 1 : ns
        for i = 1 : ng
            g{i}(j + no(i)) = pointer;
        end
        pointer = pointer+1; 
    end
end

function dg = DeepGrouping(g,a,d)
    span = a;
    remain = length(g(:));
    deta = 0;
    t = 1;
    ng = 1;
    dg = cell(1,1);
    while remain > span + a
        for i = 1 : span
            dg{ng}(i) = g(t);
            t = t + 1;
        end
        remain = remain - span;
        deta = deta + d;
        span = span + deta;
        ng = ng + 1;
    end
    if remain > 0
        for i = 1 : remain
            dg{ng}(i) = g(t);
            t = t + 1;
        end
    end
end

function dg = UniformGrouping(g,k)
    remain = mod(length(g(:)),k);
    divisor = floor(length(g(:))./k);
    t = 1;
    dg = cell(1,1);
    for i = 1 : k
        for j = 1 : divisor
            dg{i}(j) = g(t);
            t = t + 1;
        end 
    end
    for i = 1 : remain
        dg{i}(divisor+1) = g(t);
        t = t + 1;
    end
end
