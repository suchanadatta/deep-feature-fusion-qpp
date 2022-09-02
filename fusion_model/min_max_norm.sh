if [ ! -d data/interaction_letor_hist_pp/ ]
then  
mkdir data/interaction_letor_hist_pp/
fi

for f in `find data/interaction_letor_hist -type f`
do
filename=$(basename -- "$f")
echo $filename

cat data/interaction_letor_hist/$filename | awk '{for(i=1;i<=123;i++) printf("%s ",$i); for(i=124;i<=NF;i++) s+=$i; for(i=124;i<=NF;i++) {if (s==0) a=0; else a=$i/s; printf("%.4f ",a); } printf("\n")}' > data/interaction_letor_hist_pp/$filename
done
