WORDS="coach education execution figure job letter match mission mood paper post pot range rest ring scene side soil strain test"

for WORD in $WORDS; do
	echo perl scorer.pl baseline_results/"$WORD".txt wsd2-master/data/test/fr/"$WORD"_gold.txt -t best
	perl scorer.pl baseline_results/"$WORD".txt wsd2-master/data/test/fr/"$WORD"_gold.txt -t best
done
