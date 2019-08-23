#1/bin/bash

CLASSPATH=$CLASSPATH:/users5/kliao/working/tools/ollie-app-latest.jar
CLASSPATH=$CLASSPATH:.

INPUT_BASE=/users5/kliao/working/data/nyt_raw_text
OUTPUT_BASE=/users5/kliao/working/data/nyt_ollie

export JAVA_OPTS="-Xmx10g"

# compile OpenExtract.scala
scalac -classpath $CLASSPATH OpenExtract.scala

for year in $@; do
    scala -classpath $CLASSPATH ollie.OpenExtract $INPUT_BASE/$year $OUTPUT_BASE/$year.txt
done
