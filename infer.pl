:-include('out/gpt_what_kind_of_internal_logic_(classical,_intuitionistic,_non-.pro').


%% transitive closure for paths of length 1 to 4      
tc_fact(S,[V1,V2],O):-
  fact(S,V1,X),
  fact(X,V2,O).
tc_fact(S,[V1,V2,V3],O):-
  fact(S,V1,X),
  fact(X,V2,Y),
  fact(Y,V3,O).  
tc_fact(S,[V1,V2,V3,V4],O):-
  fact(S,V1,X),
  fact(X,V2,Y),
  fact(Y,V3,Z),
  fact(Z,V4,O).
   
%% get paths where all verbs are unique
tc_facts_with_unique_verbs(S,SortedVs,O):-
    tc_fact(S,Vs,O),
    S\=O,
    sort(Vs,SortedVs).


%% get inferred facts with given count of distinct verbs
inferred_with(DistinctVerbCount,S,Vs,O):-
  distinct(tc_facts_with_unique_verbs(S,Vs,O)),
  length(Vs,DistinctVerbCount),
  \+generated_fact(S,Vs,O).

%% get all directly generated facts
generated_fact(S,[V],O):-
  distinct(fact(S,V,O)).

%% get all facts with a specific number of verbs
all_facts(_,S,Vs,O):-
  generated_fact(S,Vs,O).
all_facts(VerbCount,S,Vs,O):-
  inferred_with(VerbCount,S,Vs,O).
 
%% count distinct facts
count_facts(Pred,Args,Count):-
  Callable=..[Pred|Args],
  findall(Args,distinct(Args,Callable),Xs),
  length(Xs,Count).

%% count facts in each predicate
count(N):-
   count_facts(generated_fact,[_,_,_],Count1),
   write(generated_fact:Count1),nl,
   count_facts(inferred_with,[N,_,_,_],Count2),
   write(inferred:Count2),nl,
   count_facts(all_facts,[N,_,_,_],Count3),  
   write(all_facts:Count3),nl.

%% print all facts with N distinct verbs
query(N,S):-
   all_facts(N,S,Vs,O),
   write((S,Vs,O)),nl,
   fail.

%% print all inferred facts with N distinct verbs
inf_query(N):-
  inferred_with(N,S,Vs,O),
  write((S,Vs,O)),nl,
  fail.  

%% main entry point - runs several tests
go:-
  member(S,[reasoning_llm_output,inconsistency_tolerance,paraconsistent_logic]),
     between(1,4,N),
       nl,write('--- Facts with '),write(S),write(' verbs:'),write(N),nl,   
       query(N,S).