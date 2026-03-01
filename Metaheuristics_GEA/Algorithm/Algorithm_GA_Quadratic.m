function [Ans, BestCost, pop, time, Contribution_Rate]=Algorithm_GA_Quadratic(Solution, Info, Instraction)

global NFE;
NFE=0;

%% initialization
costfunction=@(q) Cost(q,Info.Model);       % Cost Function
NCrossover=2*round((Info.PCrossover*Info.Npop)/2);
NMutation=floor(Info.PMutation*Info.Npop);

NCrossover_Scenario=floor(Info.NCrossover_Scenario*(Info.PScenario3*Info.Npop));
NMutate_Scenario=floor(Info.NMutation_Scenario*(Info.PScenario3*Info.Npop));

BestCost=zeros(Info.Iteration,1);

individual.Position1=[];
individual.Xij=[];
individual.Cost=[];
individual.CVAR=[];
pop=repmat(individual,Info.Npop,1);
popc=repmat(individual,NCrossover,1);
popm=repmat(individual,NMutation,1);

if(Instraction(1))
    temp_Scens=repmat(individual,NCrossover_Scenario*2,1);
elseif(Instraction(2))
    temp_Scens=repmat(individual,NMutate_Scenario,1);
elseif(Instraction(3))
    temp_Scens=repmat(individual,NMutate_Scenario,1);
end

if (Instraction(1) && Instraction(2) && Instraction(3))
    temp_Scens=repmat(individual,(NCrossover_Scenario*2)+NMutate_Scenario+NMutate_Scenario, 1);
end


% BestSol=Solution;

[BestSol, ~] = find(Solution.Solution);
BestSol = reshape(BestSol, 1, size(BestSol,1));  % Convert the 2D solution to 1D chromosome
%% Initial population

pop(1).Position1 = BestSol;
[pop(1).Position1, pop(1).Xij] = CreateXij(pop(1).Position1, Info.Model);
[pop(1).Cost, pop(1).Xij, pop(1).CVAR]=CostFunction(pop(1).Xij, Info.Model);

for i=2:Info.Npop   

    pop(i).Position1 = Mutation(pop(1).Position1 ,Info.Model);
    % Create Xij
    [pop(i).Position1, pop(i).Xij] = CreateXij(pop(i).Position1, Info.Model);
    % Evaluation
    [pop(i).Cost, pop(i).Xij, pop(i).CVAR]=CostFunction(pop(i).Xij, Info.Model);

    while pop(i).Cost == inf
        pop(i).Position1 = Mutation(pop(1).Position1 ,Info.Model);
        % Create Xij
        [pop(i).Position1, pop(i).Xij] = CreateXij(pop(i).Position1, Info.Model);
        % Evaluation
        [pop(i).Cost, pop(i).Xij, pop(i).CVAR]=CostFunction(pop(i).Xij, Info.Model);
    end
end

% Sort Population
Costs=[pop.Cost];
[Costs, SortOrder]=sort(Costs);
pop=pop(SortOrder);
pop=pop(1:Info.Npop);

% Store Cost
BestSol=pop(1);
WorstCost=pop(end).Cost;
beta=10;         % Selection Pressure (Roulette Wheel)

tic;
%% GA Main loop
for It=1:Info.Iteration
    
    % Probability for Roulette Wheel Selection
    P=exp(-beta*Costs/WorstCost);
    P=P/sum(P);
    
    %% Crossover
    for k=1:2:NCrossover

        i1=RouletteWheelSelection(P);
        i2=RouletteWheelSelection(P);

        pop_(1)=pop(i1);
        pop_(2)=pop(i2);

        [popc(k).Position1, popc(k+1).Position1]=Crossover(pop_, Info.Model);
        
        % Create Xij for new offspring
        [popc(k).Position1, popc(k).Xij] = CreateXij(popc(k).Position1, Info.Model);
        [popc(k+1).Position1, popc(k+1).Xij] = CreateXij(popc(k+1).Position1, Info.Model);

        % evaluate
        [popc(k).Cost, popc(k).Xij, popc(k).CVAR]=CostFunction(popc(k).Xij, Info.Model);

        [popc(k+1).Cost, popc(k+1).Xij, popc(k+1).CVAR]=CostFunction(popc(k+1).Xij, Info.Model);
    end
    
    %% Mutation
    for k=1:NMutation
        popm(k).Position1=Mutation(pop(randsample(1:Info.Npop,1)).Position1 ,Info.Model);

        % Create Xij
        [popm(k).Position1, popm(k).Xij] = CreateXij(popm(k).Position1, Info.Model);

        % Evaluation
        [popm(k).Cost, popm(k).Xij, popm(k).CVAR]=CostFunction(popm(k).Xij, Info.Model);

    end
    

    %% scenario 1 : Dominated Gen
    if(Instraction(1))
        [DominantGenes,Mask, DominantChromosome,Mask_Dominant]=Analyze_Perm(pop(1:(Info.PScenario1*Info.Npop)),Info);
        Mask;
        for k=1:2:NCrossover_Scenario*2
            i1=RouletteWheelSelection(P);
            pop__(1)=DominantChromosome;
            pop__(2)=pop(i1);
            [temp_Scens(k).Position1 temp_Scens(k+1).Position1]=Crossover(pop__, Info.Model);

            % Create Xij for new offspring
            [temp_Scens(k).Position1, temp_Scens(k).Xij] = CreateXij(temp_Scens(k).Position1, Info.Model);
            [temp_Scens(k+1).Position1, temp_Scens(k+1).Xij] = CreateXij(temp_Scens(k+1).Position1, Info.Model);

            % evaluate
            [temp_Scens(k).Cost, temp_Scens(k).Xij, temp_Scens(k).CVAR]=CostFunction(temp_Scens(k).Xij, Info.Model);
            [temp_Scens(k+1).Cost, temp_Scens(k+1).Xij, temp_Scens(k+1).CVAR]=CostFunction(temp_Scens(k+1).Xij, Info.Model);
        end
    end
    
    %% scenario 2 : mask mutation in goods
    L=0;
    if(Instraction(2))
        [~,Mask,~,~]=Analyze_Perm(pop(1:(Info.PScenario2*Info.Npop)),Info);
        for i=1:NMutate_Scenario
            ii = randsample(1:(Info.PScenario2*Info.Npop),1);
            temp_Scens(L+i).Position1 = MaskMutation(Info.MaskMutationIndex,pop(ii).Position1,Mask(ii,:),Info.Model);
            
            % Create Xij for new offspring
            [temp_Scens(L+i).Position1, temp_Scens(L+i).Xij] = CreateXij(temp_Scens(L+i).Position1, Info.Model);

            % evaluate
            [temp_Scens(L+i).Cost, temp_Scens(L+i).Xij, temp_Scens(L+i).CVAR]=CostFunction(temp_Scens(L+i).Xij, Info.Model);
        end
    end
    
    %% scenario 3 : inject good gens
    L=0;
    if(Instraction(3))
        [DominantGenes,Mask,~, Mask_Dominant]=Analyze_Perm(pop(1:(Info.PScenario3*Info.Npop)),Info);
        for z=1:NMutate_Scenario
            jj = randsample(size(pop,1)-(Info.PScenario3*Info.Npop):size(pop,1),1);
            temp_Scens(L+z).Position1 = CombineQ(DominantGenes.Position1.Position1,pop(jj).Position1,Mask_Dominant,Info.Model);
                        
            % Create Xij for new offspring
            [temp_Scens(L+z).Position1, temp_Scens(L+z).Xij] = CreateXij(temp_Scens(L+z).Position1, Info.Model);

            % evaluate
            [temp_Scens(L+z).Cost, temp_Scens(L+z).Xij, temp_Scens(L+z).CVAR]=CostFunction(temp_Scens(L+z).Xij, Info.Model);        
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculate the contrubution share for Sensitivity Analysis GEA_3  %%%%%%%%%%%%%%
    index_pop = [1:Info.Npop];
    index_crossover = [Info.Npop+1:Info.Npop+numel(popc)];
    index_mutation = [Info.Npop+numel(popc)+1:Info.Npop+numel(popc)+numel(popm)];

    index_Scenario_3 = [Info.Npop+numel(popc)+numel(popm)+1:Info.Npop+numel(popc)+numel(popm)+numel(temp_Scens)];
    
    %% Pool fusion & Selection Best Chromosome

    % Elitism Selection (Npop best will be selected)
    % Create Merged Population
    if (size(temp_Scens,1)>1)
        pop=[pop;popc;popm;temp_Scens];
    else
        pop=[pop;popc;popm];
    end

    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder); 
    pop=pop(1:Info.Npop);
    Costs=[pop.Cost];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Calculate share rate for all operators for GEA_3 %%%%%%%%%%%%%%%%%%%%%%%%%
    previous_pop = size(intersect(SortOrder(1:Info.Npop), index_pop), 2) / Info.Npop;
    crossover_ShareRate = size(intersect(SortOrder(1:Info.Npop), index_crossover), 2) / Info.Npop;
    mutation_ShareRate = size(intersect(SortOrder(1:Info.Npop), index_mutation), 2) / Info.Npop;
    Scenario3_ShareRate = size(intersect(SortOrder(1:Info.Npop), index_Scenario_3), 2) / Info.Npop;
    Contribution_Rate(It, :) = [previous_pop crossover_ShareRate mutation_ShareRate Scenario3_ShareRate];

    % Update Worst Cost
    WorstCost=max(WorstCost,pop(end).Cost);
    
    % Store Best Solution Ever Found
    if pop(1).Cost < BestSol.Cost
    BestSol=pop(1);
    end

    BestCost(It)=BestSol.Cost;
    BestPosition=pop(1).Position1;
    
    % Store NFE
    nfe(It)=NFE;
    
   % Show Iteration Information
    disp(['Iteration ' num2str(It)  ', Best Cost = ' num2str(BestCost(It))]);
    time = toc;
    if time>=1000
        break;
    end
    
end
time;    
Ans=pop(1).Cost;
Solution=BestSol; 
end