package gov.sandia.dfgleic;

import Jama.Matrix;
import au.com.bytecode.opencsv.CSVReader;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.mapred.lib.IdentityMapper;
import org.apache.hadoop.typedbytes.Type;
import org.apache.hadoop.typedbytes.TypedBytesInput;
import org.apache.hadoop.typedbytes.TypedBytesOutput;
import org.apache.hadoop.typedbytes.TypedBytesWritable;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import Jama.*;
import java.io.*;
import java.util.*;


import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.mapred.lib.IdentityMapper;
import org.apache.hadoop.typedbytes.Type;
import org.apache.hadoop.typedbytes.TypedBytesInput;
import org.apache.hadoop.typedbytes.TypedBytesOutput;
import org.apache.hadoop.typedbytes.TypedBytesWritable;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import weka.classifiers.Classifier;
import weka.core.*;

import java.util.*;
import java.io.*;
import java.lang.*;


/**
 * Created with IntelliJ IDEA.
 * User: hubiquitus
 * Date: 3/27/15
 * Time: 4:02 PM
 * To change this template use File | Settings | File Templates.
 */
public class Bagging_MR extends Configured implements Tool {
    private static final Logger sLogger = Logger.getLogger(Bagging_MR .class);



    public static void main(String args[]) throws Exception {
        // Let ToolRunner handle generic command-line options
        int res = ToolRunner.run(new Configuration(), new Bagging_MR(), args);

        System.exit(res);
    }

    private static int printUsage() {
        System.out.println("usage: -mat <filepath>  [-output <outputpath>]\n" +
                "  [-block_size <int>] [-split_size <int>] [-mem <int]");
        ToolRunner.printGenericCommandUsage(System.out);
        return -1;
    }

    private String getArgument(String arg, String[] args) {
        for (int i=0; i<args.length; ++i) {
            if (arg.equals(args[i])) {
                if (i+1<args.length) {
                    return args[i+1];
                } else {
                    return null;
                }
            }
        }
        return null;
    }


    public static String removeExtension(String s) {

        String separator = System.getProperty("file.separator");

        // Remove the extension.
        int extensionIndex = s.lastIndexOf(".");
        if (extensionIndex == -1)
            return s;

        return s.substring(0, extensionIndex);
    }

    public static String getExtension(String s) {

        String separator = System.getProperty("file.separator");

        // Get the extension.
        int extensionIndex = s.lastIndexOf(".");
        if (extensionIndex == -1)
            return s;

        return s.substring(extensionIndex+1);
    }

    public int run(String[] args) throws Exception {

        if (args.length == 0) {
            return printUsage();
        }


        String matfile = getArgument("-mat",args);
        if (matfile == null) {
            System.out.println("Required argument '-mat' missing");
            return -1;
        }

        String ext=getExtension(matfile);
        String base=removeExtension(matfile);

        String outputfile = getArgument("-outputttttttttt",args);
        if (outputfile == null) {
            outputfile = base + "-Result." + ext;
        }

        String reduceSchedule = getArgument("-reduce_schedule",args);
        if (reduceSchedule == null) {
            reduceSchedule = "1";
        }

        String block_Size = getArgument("-block_size",args);
        if (block_Size == null) {
            block_Size = "1";
        }

        String PathTest = getArgument("-test_file",args);
        if (PathTest == null) {
            System.out.println("Required argument '-test_file' missing");
            return -1;
        }

        String nbBagIter = getArgument("-Bag_Iter",args);
        if (nbBagIter == null) {
            System.out.println("Required argument '-Bag_Iter' missing");
            return -1;
        }
        String nbTest = getArgument("-nbTest",args);
        if (nbTest == null) {
            System.out.println("Required argument '-nbTest' missing");
            return -1;
        }

        String splitSize = getArgument("-split_size",args);

        sLogger.info("Tool name: Bagging_MR");
        sLogger.info(" -mat: " + matfile);
        sLogger.info(" -output: " + outputfile);
        sLogger.info(" -reduce_schedule: " + reduceSchedule);
        sLogger.info(" -block_size: " + block_Size);
        sLogger.info(" -test_file: " + PathTest);
        sLogger.info(" -Bag_Iter: " + nbBagIter);
        sLogger.info(" -nbTest: " + nbTest);
        sLogger.info(" -split_size: " +
                (splitSize == null ? "[Default]" : splitSize));

        String stages[] = reduceSchedule.split(",");
        String curinput = matfile;
        String curoutput = outputfile;


        for (int stage=0; stage<stages.length; ++stage) {
            int numReducers = Integer.parseInt(stages[stage]);

            if (stage > 0) {
                curinput = curoutput;
            }

            if (stage+1 < stages.length) {
                curoutput = outputfile + "_iter"+(stage+1);
            } else {
                curoutput = outputfile;
            }

            // run the iteration
            // TODO make this a separate function?
            JobConf conf = new JobConf(getConf(), Bagging_MR.class);
            DistributedCache.createSymlink(conf);
            conf.setJobName(
                    "Bagging_MR.java (" + (stage+1) + "/" + stages.length + ")");

            conf.setNumReduceTasks(numReducers);
            conf.setInt("block_size", Integer.parseInt(block_Size));
            conf.setInt("stage", stage);
            //conf.set("mapred.child.java.opts","-Xmx2G");
            if (splitSize != null) {
                conf.set("mapred.minsplit.size", splitSize);
                conf.set("mapreduce.input.fileinputformat.split.minsize", splitSize);
            }


            //partage du fichier de test
            conf.set("test_path",PathTest);
            conf.set("nbTest",nbTest);
            //partage du nb iteration du bagging

            conf.set("Bag_Iter",nbBagIter);
            // set the formats
            conf.setInputFormat(SequenceFileInputFormat.class);
            conf.setOutputFormat(SequenceFileOutputFormat.class);

            // set the data types
            conf.setOutputKeyClass(Text.class);
            conf.setOutputValueClass(TypedBytesWritable.class);

            if (stage ==1) {
                FileInputFormat.setInputPaths(conf, new Path(curinput));

                conf.setMapperClass(IdentityMapper.class);
                //conf.setReducerClass(TSQRReducer2.class);
            }
            else if (stage==2)
            {   FileInputFormat.setInputPaths(conf, new Path(curinput));

                conf.setMapperClass(IdentityMapper.class);
                // conf.setReducerClass(TSQRReducer3.class);
            } else {
                FileInputFormat.setInputPaths(conf, new Path(curinput));

                Path outputPath = new Path(args[3]);

                conf.setMapperClass(BaggingMapper.class);

                conf.setReducerClass(BaggingReducer.class);
            }


            FileOutputFormat.setOutputPath(conf, new Path(curoutput));

            sLogger.info("Iteration " + (stage+1) + " of " + stages.length);
            sLogger.info(" - reducers: " + numReducers);
            sLogger.info(" - curinput: " + curinput);
            sLogger.info(" - curoutput: " + curoutput);

            JobClient.runJob(conf);

        }

        return 0;
    }

    public static class BaggingIteration
            extends MapReduceBase
    {
        boolean is_reduce = false;
        protected DenseMatrix mat_Test;

        String TestPath;
        protected int nbTest;
        protected int nbBagIter;
        protected int blockSize;
        protected int numColumns;
        protected int currentRow;
        protected DenseVector Y_reel;
        protected DenseVector Somm_Y_ALL;
        Double Agregate;
        CSVReader products;
        int nbModel;
        double[] Y_test;
        Long row_id;
        ArrayList<Long> row_ids;
        protected Random rand;
        DenseMatrix A;
        double [] Y_A=null;
        DenseMatrix A1;
        DenseMatrix Q1;
        DenseMatrix Q2;
        DenseMatrix Q_int;
        DenseMatrix Q3;
        DenseMatrix R;
        DenseMatrix R_int;
        DenseVector Val;
        boolean flag=false;
        boolean flagY=false;
        boolean NVide= false;
        Classifier Y_init=null;
        double[]Yi=null;
        String [] argv;
        double[]Ytransit=null;
        boolean isFirstIteration = false;
        int numReduce = 0;
        /** weak classifiers */
        private Classifier[] classifiers;

        /** weights for all weak classifiers */
        private double[] cweights;
        int i1=0,j1=0;
        // this output must be set at some point before close,
        // if there is going to be any output.
        protected OutputCollector<Text,TypedBytesWritable> output;

        public BaggingIteration() {
            this.currentRow = 0;
            this.A = null;
            this.output = null;
            this.rand = new Random();
        }

        public BaggingIteration (int blockSize) {
            this();
            this.blockSize = blockSize;
        }



        public Instances createDataset1(DenseMatrix A) throws Exception {
            FastVector atts = new FastVector();
            List<Instance> instances = new ArrayList<Instance>();
            for(int dim = 0; dim < A.numColumns(); dim++)
            {
                Attribute current = new Attribute("Attribute" + dim, dim);

                for(int obj = 0; obj < A.numRows(); obj++)
                {
                    if(dim == 0)
                    {
                        instances.add(new SparseInstance(A.numRows()));
                    }
                    instances.get(obj).setValue(current, A.get(obj,dim));
                }

                atts.addElement(current);
            }

            Instances newDataset =  new Instances("Dataset1", atts, instances.size());

            for(Instance inst : instances)
                newDataset.add(inst);
            return newDataset;

        }

        protected TypedBytesWritable randomKey() throws IOException {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();

            TypedBytesOutput out =
                    new TypedBytesOutput(new DataOutputStream(bytes));
            out.writeInt(rand.nextInt(2000000000));

            TypedBytesWritable val = new TypedBytesWritable(bytes.toByteArray());

            return val;
        }

        protected TypedBytesWritable encodeTypedBytes(double array[])
                throws IOException {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();

            TypedBytesOutput out =
                    new TypedBytesOutput(new DataOutputStream(bytes));

            out.writeVectorHeader(array.length);
            for (int i=0; i<array.length; ++i) {
                out.writeDouble(array[i]);
            }

            TypedBytesWritable val =
                    new TypedBytesWritable(bytes.toByteArray());

            return val;
        }

        protected DenseMatrix Clean_matrix (DenseMatrix Q)
        {
            DenseMatrix result;
            boolean Zero = true;
            List<double[]> rows = new ArrayList<double[]>();
            for(int i=0;i<Q.numRows();i++)
            {
                double[] row = new double[Q.numColumns()];
                Zero = true;
                for (int j=0;j<Q.numColumns();j++)
                {
                    if (Q.get(i,j) != 0)
                        Zero = false;

                    row[j] = Q.get(i,j);
                }
                if (!Zero)
                    rows.add(row);
            }
            result = new DenseMatrix(rows.size(),Q.numColumns());
            for(int i=0;i<result.numRows();i++)
            {
                for (int j=0;j<result.numColumns();j++)
                    result.set(i,j,rows.get(i)[j]);
            }

            return result;
        }

        protected TypedBytesWritable encodeTypedBytesMatrix(DenseMatrix D)
                throws IOException {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();

            TypedBytesOutput out =
                    new TypedBytesOutput(new DataOutputStream(bytes));
            List<ArrayList<Double>> MatrixList = new ArrayList<ArrayList<Double>>(D.numRows());

            for (int i=0; i<D.numRows(); ++i) {
                ArrayList<Double> Raw = new ArrayList<Double>(D.numColumns());
                for (int j=0; j<D.numColumns(); ++j) {
                    Raw.add(D.get(i,j));
                }
                MatrixList.add(Raw);

            }
            out.writeList(MatrixList);

            TypedBytesWritable val =
                    new TypedBytesWritable(bytes.toByteArray());

            return val;
        }

        double readDouble(TypedBytesInput in, Type t) throws IOException {
            if (t == Type.BOOL) {
                boolean b = in.readBool();
                if (b == true) {
                    return 1.;
                } else {
                    return 0.;
                }
            } else if (t == Type.BYTE) {
                byte b = in.readByte();
                return (double)b;
            } else if (t == Type.INT) {
                int i = in.readInt();
                return (double)i;
            } else if (t == Type.LONG) {
                long l = in.readLong();
                return (double)l;
            } else if (t == Type.FLOAT) {
                float f = in.readFloat();
                return (double)f;
            } else if (t == Type.DOUBLE) {
                return in.readDouble();
            } else {
                throw new IOException("Type " + t.toString() + " cannot be converted to double ");
            }
        }

        protected double[] doubleArrayListToArray(ArrayList<Double> a) {
            double rval[] = new double[a.size()];
            for (int i=0; i<a.size(); ++i) {
                rval[i] = a.get(i).doubleValue();
            }
            return rval;
        }

        protected double[] decodeTypedBytesArray(TypedBytesWritable bytes)
                throws IOException {

            TypedBytesInput in =
                    new TypedBytesInput(
                            new DataInputStream(
                                    new ByteArrayInputStream(bytes.getBytes())));

            Type t = in.readType();
            if (t == Type.VECTOR || t == Type.LIST) {
                if (t == Type.VECTOR) {
                    ArrayList<Double> d = new ArrayList<Double>();
                    int len = in.readVectorHeader();
                    for (int i=0; i<len; ++i) {
                        Type et = in.readType();
                        d.add(new Double(readDouble(in, et)));
                    }
                    return doubleArrayListToArray(d);
                } else {
                    ArrayList<Double> d = new ArrayList<Double>();
                    while (true) {
                        Type et = in.readType();
                        if (et == Type.MARKER) {
                            break;
                        }
                        d.add(new Double(readDouble(in, et)));
                    }
                    return doubleArrayListToArray(d);
                }
            } else {
                return null;
            }
        }
        protected DenseMatrix decodeTypedBytesMatrix(TypedBytesWritable bytes)
                throws IOException {

            TypedBytesInput in =
                    new TypedBytesInput(
                            new DataInputStream(
                                    new ByteArrayInputStream(bytes.getBytes())));

            Type t = in.readType();
            if (t == Type.LIST) {

                ArrayList<ArrayList<Double>> l = new ArrayList<ArrayList<Double>>();
                l = (ArrayList<ArrayList<Double>>) in.readList();
                DenseMatrix D = new DenseMatrix(l.size(),l.get(0).size());
                for (int i =0;i < l.size(); i++ )
                {
                    ArrayList<Double> a = l.get(i);
                    for (int j=0;j<a.size();j++)
                        D.set(i,j,a.get(j));
                }
                return D;
            } else {
                return null;
            }
        }


          public DenseVector Calcul_Param(DenseMatrix OutWeight)     throws IOException
          {

              //recuperation du fichier de test
              this.products = new CSVReader(new FileReader(this.TestPath));



              String[] nextLine;
              boolean send_Y_reel = false;
              DenseVector Y_Prob = new DenseVector(OutWeight.numRows());
              DenseVector Y_SOMM = new DenseVector(nbTest);
              if (Y_reel == null || Y_reel.size() == 0)
              {
                  Y_reel = new DenseVector(nbTest);
                  send_Y_reel = true;
              }
              try {
                  int iLine = 0;
                  while ((nextLine = products.readNext()) != null)
                  {
                    DenseVector XTest_V = new DenseVector(nextLine.length);
                    Double Y = null;
                    for (int i=0;i<nextLine.length;i++)
                    {
                        if (i==nextLine.length-1)
                        {
                            Y = Double.parseDouble(nextLine[i]);
                            XTest_V.set(i,1);
                            
                            if (send_Y_reel)
                                Y_reel.add(iLine, Y);
                        }
                        else
                        XTest_V.set(i,Double.parseDouble(nextLine[i]));

                        
                    }
                    OutWeight.mult(XTest_V,Y_Prob);
                    Double Somm_Y_Prob = 0.0;
                    for (int i=0;i<Y_Prob.size();i++)
                    {
                         Somm_Y_Prob = Somm_Y_Prob + Y_Prob.get(i); //(y1+y2/|S|-y')2
                    }
                   Y_SOMM.set(iLine, Somm_Y_Prob);

                   iLine++;
                  }

                  Text KeyY = new Text();
                  KeyY.set("Y_SOMM_".toString().concat(String.valueOf(row_id)));
                  output.collect(KeyY, encodeTypedBytes(Y_SOMM.getData()));
                  //Envoyer Y_Reel une seule fois
                  if (send_Y_reel) //Envoyer Y_Reel une seule fois
                  {
                      KeyY = new Text();
                      KeyY.set("Y_Reel".toString());
                      output.collect(KeyY, encodeTypedBytes(Y_reel.getData()));
                  }
                  Y_reel = null;


              } catch (IOException e) {
                  e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
              }
              return Y_SOMM;
          }


        public DenseMatrix Compress(DenseMatrix A) throws Exception {


            Instances datamoufi = new Instances(createDataset1(A));
            datamoufi.setClassIndex(datamoufi.numAttributes()-1);
            Bagging B=new Bagging ();
            // runClassifier(B, argv);
            String options = String.format(" java weka.classifiers.meta.ClassificationViaRegression -W weka.classifiers.functions.LinearRegression \\\n" +
                    " -x 2 -I %d -- -S 1", nbBagIter);
            String[] optionsArray = options.split(" ");

                B.setOptions(optionsArray);
                B.setCalcOutOfBag(true);
                B.buildClassifier(datamoufi);


              return B.Output_weight2(A.numColumns());
        }
        public void collect2(TypedBytesWritable key, TypedBytesWritable value)
                throws Exception
        {
            //Apprentissage
            double row[] = decodeTypedBytesArray(value);

            if (currentRow == 0)
                row_id = Long.parseLong(key.getValue().toString());


            if (A == null)
            {
                numColumns = row.length-1;
                A = new DenseMatrix((numColumns-1)*blockSize,numColumns);  //car le A contient la colonne y qui ne doit pas rentrer dans le calcul du blocksize

            }

            for (int i=1; i<row.length; i++) {
                    A.set(currentRow, i-1, row[i]);
            }
            currentRow ++;

            if (currentRow >= A.numRows()) {
                currentRow=0;
                Calcul_Param(Compress(A)); //Column correspond nb modele = iteration du classifier, ligne correspond au nb de test
                //DenseMatrix M = Compress(A);
                //output.collect(new Text("B"),encodeTypedBytesMatrix(M));
                A = null;
            }
        }
        public void close() throws IOException
        {

            if (output != null)
            {
                if (is_reduce) {

                    Agregate = 0.0;
                    for (int j=0;j<Somm_Y_ALL.size();j++)
                    {
                       Agregate = Agregate + (Somm_Y_ALL.get(j)/nbModel - Y_reel.get(j)) * (Somm_Y_ALL.get(j)/nbModel - Y_reel.get(j));
                    }
                    Double MSE = Agregate/nbTest;
                    MSE = Math.sqrt(MSE);
                    Text keyMSE =new Text();
                    keyMSE.set("MSE".toString()) ;
                    double[] MSE_param=new double[1] ;
                    MSE_param[0]=MSE;
                    output.collect(keyMSE,encodeTypedBytes(MSE_param));
                    MSE_param[0]=nbModel;
                    output.collect(new Text("nbModel"),encodeTypedBytes(MSE_param));
                }
                else if (A != null) {

                        DenseMatrix A_Cleaned = Clean_matrix(A);
                        try {
                          Calcul_Param(Compress(A_Cleaned));
                            //DenseMatrix M = Compress(A_Cleaned);
                            //output.collect(new Text("B"),encodeTypedBytesMatrix(M));

                        } catch (Exception e) {
                            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
                        }

                }
            }
        }
    }
    public static class BaggingMapper
            extends BaggingIteration
            implements Mapper<TypedBytesWritable, TypedBytesWritable,Text, TypedBytesWritable> {

        public void configure(JobConf job){
            this.blockSize = Integer.parseInt(job.get("block_size"));
            this.isFirstIteration = (Integer.parseInt(job.get("stage")) == 0);
            this.nbBagIter = Integer.parseInt(job.get("Bag_Iter"));
            this.nbTest = Integer.parseInt(job.get("nbTest"));
            this.TestPath = job.get("test_path");
         }

        public void map(TypedBytesWritable key, TypedBytesWritable value,
                        OutputCollector<Text,TypedBytesWritable> output,
                        Reporter reporter)
                throws IOException {

            if (this.output == null) {
                this.output = output;
            }
            flag = true;



            try {
                collect2(key, value);

            } catch (Exception e) {
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }


        }
    }




    public static class BaggingReducer
            extends BaggingIteration
            implements Reducer<Text, TypedBytesWritable,Text, TypedBytesWritable> {
        public void configure(JobConf job){
            this.blockSize = Integer.parseInt(job.get("block_size"));
            this.isFirstIteration = (Integer.parseInt(job.get("stage")) == 0);
            this.Agregate = 0.0;
            this.nbTest = Integer.parseInt(job.get("nbTest"));
            Somm_Y_ALL = new DenseVector(nbTest);
            this.nbBagIter = Integer.parseInt(job.get("Bag_Iter"));


        }


        public void reduce(Text key, Iterator<TypedBytesWritable> values,
                           OutputCollector<Text,TypedBytesWritable> output,
                           Reporter reporter)
                throws IOException {
            if (this.output == null) {
                this.output = output;
            }

             is_reduce = true;
            while (values.hasNext()) {
                //output.collect(key,values.next());

                if (key.toString().startsWith("Y_SOMM"))
                {
                    double[] Y_Residual = decodeTypedBytesArray(values.next());

                    for (int j=0;j<Y_Residual.length;j++)
                    {

                         Somm_Y_ALL.set(j, Somm_Y_ALL.get(j)+ Y_Residual[j]);
                    }

                    nbModel = nbModel + nbBagIter;
                }
                else if (key.toString().equals("Y_Reel"))
                {
                    Y_reel = new DenseVector(decodeTypedBytesArray(values.next()));
                }
            }

       }
    }

}







