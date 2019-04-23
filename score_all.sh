WORDS="coach education execution figure job letter match mission mood paper post pot range rest ring scene side soil strain test"
OUTFILE="summary_results.txt"

rm -rf $OUTFILE
for WORD in $WORDS; do
	echo perl scorer.pl baseline_results/"$WORD".txt fr/"$WORD"_gold.txt -t best
	perl scorer.pl baseline_results/"$WORD".txt fr/"$WORD"_gold.txt -t best
	precision=`sed '3q;d' baseline_results/"$WORD".txt.results | awk -F',' ' {print $1} ' | awk '{print $3}'`
	echo $WORD $precision >> $OUTFILE
done

avg=`awk '{ total += $2; count++ } END { print total/count }' $OUTFILE`
echo 'AVG' $avg >> $OUTFILE
