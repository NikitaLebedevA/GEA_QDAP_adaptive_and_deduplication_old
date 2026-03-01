function x1=Mutation_Random(q,model)
    n=size(q,2);

    num_rand=randi([1 5]);
    Point=randsample(1:n-1,num_rand);
    for i=1:num_rand
        r1=randi([1 model.I]);
        q(Point(i))=r1;
    end
    x1=q;
end
