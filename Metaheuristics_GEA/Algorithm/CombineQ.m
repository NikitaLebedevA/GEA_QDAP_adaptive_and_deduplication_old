function Ans=CombineQ(Position01,Position02,Pattern,Model)
    PatternI=abs(Pattern*-1+ones(1,size(Pattern,2)));
    q=Position01.*Pattern+Position02.*PatternI;


    Ans=q;
end