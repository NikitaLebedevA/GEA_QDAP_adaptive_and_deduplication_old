function pop=ConverC2D(pop,NGap)
%Conver Continuous Choromosome to Discrete
npop=size(pop,1);
n=size(pop(1).Position,2);
Interval=1/NGap;
for i=1:npop;
    for j=1:n
        pop(i).Position(j)=floor((pop(i).Position(j))/Interval)+1;
    end
end
end