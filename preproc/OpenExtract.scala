package ollie
import io.Source
import edu.knowitall.ollie.Ollie
import edu.knowitall.tool.parse.MaltParser
import edu.knowitall.tool.postag.OpenNlpPostagger
import edu.knowitall.tool.tokenize.OpenNlpTokenizer
import java.net.URL
import edu.knowitall.tool.parse.graph.DependencyGraph
import edu.knowitall.openparse.OpenParse
import java.io.{PrintWriter, StringWriter, File}

object OpenExtract{

    def sortFile(files: Array[File]): Array[File] = {
        val extractor = "([\\d]+).*$".r
        val sorted = files.sortWith {(l, r) =>
            val extractor(lFileNumber) = l.getName
            val extractor(rFileNumber) = r.getName
            lFileNumber.toInt < rFileNumber.toInt
        }
        sorted
    }

    def main(args:Array[String]) {
        var inputDir, outputFile = ""
        inputDir = args(0) //the directory in which the files come from 
        outputFile = args(1) //File to print tuples to

        val writer = new PrintWriter(outputFile, "utf-8")
        val directory = new File(inputDir) 
        val sep = "|"
        val parser = new MaltParser(new File("engmalt.linear-1.7.mco").toURI.toURL, new OpenNlpPostagger("en-pos-maxent.bin", new OpenNlpTokenizer("en-token.bin")), None)
        val openparse = OpenParse.withDefaultModel(OpenParse.Configuration(confidenceThreshold=0.005, expandExtraction=false))
        val ollie = new Ollie(openparse)
        var years_processed = -1
        var months_processed = -1
        var days_processed = -1

        val year = directory
        sortFile(year.listFiles).foreach(month => {
            if (month.isDirectory) {
                sortFile(month.listFiles).foreach(day => {
                    if (day.isDirectory) {
                        sortFile(day.listFiles).foreach(infile => { //looking at all documents in directory 
                            val docid = year.getName() + "_" + month.getName() + "_" + day.getName() + "_" + infile.getName()

                            var sentence_id = 0 //current sentence we are on
                            //End Code Specific to NYT Directory Structure
                            Source.fromFile(infile).getLines().foreach(line => {
                                try {
                                    if(line.length() > 30) { //dont bother processing short sentences
                                        val sent = parser.dependencyGraph(line)
                                        val instances = ollie.extract(sent)

                                        var event_id = 0
                                        for (instance <- instances) {
                                            val arg1 = instance.extr.arg1.text
                                            val arg2 = instance.extr.arg2.text
                                            val rel = instance.extr.rel.text

                                            val outstr = docid + sep + sentence_id + sep + arg1 + sep + rel + sep + arg2 + sep + line 
                                            println("Years: %s, Months: %s, Days: %s, Doc: %s, Sent: %s, Event: %s".format(year.getName(), month.getName(), day.getName(), infile.getName(), sentence_id, event_id))
                                            event_id = event_id + 1

                                            writer.println(outstr)
                                        }
                                    }
                                } catch {
                                    case e: Exception => System.err.println("%s, %s, %s, %s, error on sent: %s".format(year.getName(), month.getName(), day.getName(), infile.getName(), line))
                                }
                            //end for iterating over tuples
                            
                            sentence_id += 1

                            }) // end for iterating over each line

                        })  // end iterating over files (for one day)
                    } // if day is dir
                }) // days
            } // if month is dir
        }) // months

        writer.close
    }  // main
} 



