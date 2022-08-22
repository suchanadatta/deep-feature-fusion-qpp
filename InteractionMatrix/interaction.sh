#!/bin/bash
# generates interaction matrices between query and pseudo-relevant documents (for test split)
# generates interaction matrices between query and qrels (for train split)

if [ $# -le 9 ] 
then
    echo "Usage: " $0 " <following arguments in the order>";
    echo "1. Query file (in .xml format).";
    echo "2. Path of the lucene index."
    echo "3. Stopwords file."
    echo "4. SimilarityFunction: 0.DefaultSimilarity, 1.BM25Similarity, 2.LMJelinekMercerSimilarity, 3.LMDirichletSimilarity.";
    echo "5. No. of top documents to retrieve.";
    echo "6. Path of the directory to store initial retrieved documents.";
    echo "7. Word vector file path.";
    echo "8. Name of the field used for searching (default 'content'- if using available index with this project).";
    echo "9. Interaction matrix path.";
    echo "10. Qrel file path.";
    exit 1;
fi

queryPath=`readlink -f $1`
indexPath=`readlink -f $2`
stopFilePath=`readlink -f $3`
numHits=$5
retFilePath=`readlink -f $6`      # absolute directory path of the .res file
retFilePath=$retFilePath"/"
wordVecPath=`readlink -f $7`
searchField=$8
interMatrixPath=`readlink -f $9`
interMatrixPath=$interMatrixPath"/"
qrelPath=`readlink -f $10`

echo "Using query file at: "$queryPath
echo "Using index at : "$indexPath
echo "Using stop file at : "$stopFilePath
echo "Store initial retrieved file at : "$retFilePath
echo "Using word2vec file at : "$wordVecPath 
echo "Store interaction matrices at : "$interMatrixPath
echo "Qrel data used from : "$qrelPath

similarityFunction=$4

case $similarityFunction in
    1) param1=0.7
       param2=0.3 ;;
    2) param1=0.6
       param2=0.0 ;;
    3) param1=1000
       param2=0.0 ;;
esac

echo "similarity-function: "$similarityFunction" " $param1

# making the .properties file
cat > interaction.properties << EOL

queryPath=$queryPath

indexPath=$indexPath

stopFilePath=$stopFilePath

similarityFunction=$similarityFunction

param1=$param1
param2=$param2

numHits=$numHits

retFilePath=$retFilePath

wordVecPath=$wordVecPath

searchField=$searchField

interMatrixPath=$interMatrixPath

qrelPath=$qrelPath

EOL
# .properties file made

# create matching histogram for preranked documents
java -Xmx3g -cp $CLASSPATH:dist/InteractionMatrix.jar interactionmatrix.GenerateHistogramPrerankByQid
# java -Xmx3g -cp $CLASSPATH:dist/InteractionMatrix.jar interactionmatrix.GenerateHistogramPrerankFile

# create matching histogram for judged documents
#java -Xmx3g -cp $CLASSPATH:dist/InteractionMatrix.jar interactionmatrix.GenerateHistogramQrelFile
