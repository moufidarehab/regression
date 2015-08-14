package gov.sandia.dfgleic;

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

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.lang.*;


/**
 * Created with IntelliJ IDEA.
 * User: hubiquitus
 * Date: 9/19/14
 * Time: 9:23 AM
 * To change this template use File | Settings | File Templates.
 */





public class  Reseau_neurone_MR extends Configured implements Tool {
    private static final Logger sLogger = Logger.getLogger(Reseau_neurone_MR .class);



    public static void main(String args[]) throws Exception {
        // Let ToolRunner handle generic command-line options
        int res = ToolRunner.run(new Configuration(), new Reseau_neurone_MR (), args);

        System.exit(res);
    }

    private static int printUsage() {
        System.out.println("usage: -mat <filepath> [-output <outputpath>]\n" +
                "  [-maxIterations <int>] [-Hidden_Layer <int>] [-MaxError <double>] [-LearningRate <double>]");
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
            outputfile = base + "-qrr_update." + ext;
        }

        String Hidden_Layer = getArgument("-Hidden_Layer",args);
        if (Hidden_Layer == null) {
            System.out.println("Required argument '-Hidden_Layer' missing");
        }

        String maxIterations = getArgument("-maxIterations",args);
        if (maxIterations == null) {
            maxIterations = "1000";
        }

        String MaxError = getArgument("-MaxError",args);
        if (MaxError == null) {
            System.out.println("Required argument '-MaxError' missing");
        }

        String reduceSchedule = getArgument("-reduce_schedule",args);
        if (reduceSchedule == null) {
            reduceSchedule = "1";
        }

        String LearningRate = getArgument("-LearningRate",args);
        if (LearningRate == null) {
            System.out.println("Required argument '-LearningRate' missing");
        }

        sLogger.info("Tool name: Reseaux_Neurones_MR");
        sLogger.info(" -mat: " + matfile);
        sLogger.info(" -output: " + outputfile);
        sLogger.info(" -Hidden_Layer: " +Hidden_Layer );
        sLogger.info(" -maxIterations: " + maxIterations);
        sLogger.info(" -MaxError: " + MaxError);
        sLogger.info(" -LearningRate: " + LearningRate);

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
            JobConf conf = new JobConf(getConf(), Reseau_neurone_MR.class);
            DistributedCache.createSymlink(conf);
            conf.setJobName(
                    "TSQR_UPDATE.java (" + (stage+1) + "/" + stages.length + ")");

            conf.setNumReduceTasks(numReducers);
            conf.setInt("maxIterations", Integer.parseInt(maxIterations));
            conf.setInt("Hidden_Layer", Integer.parseInt(Hidden_Layer));
            //conf..setDouble("MaxError", Double.parseDouble(MaxError));
            //conf.setD("LearningRate", Float.parseFloat(MaxError));
            conf.setInt("stage", stage);
            //conf.set("mapred.child.java.opts","-Xmx2G");


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

                conf.setMapperClass(TSQRMapper.class);

                conf.setReducerClass(TSQRReducer.class);
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

    public static class TSQRIteration
            extends MapReduceBase
    {
        static int Hidden_Layer;
        protected int maxIterations;
        protected double MaxError;
        protected double LearningRate;

        protected Double numColumns;
        protected int currentRow;
        int row_id = 0;
        List<Integer> row_ids;
        protected Random rand;
        DenseMatrix A;
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
        double[]Y_init=null;
        boolean isFirstIteration = false;
        int numReduce = 0;
        int i1=0,j1=0;
         int patNum;
        double errThisPat;
        double outPred;
        double[][] trainInputs=null;
        double[] Wi_output=null;
        double[][] weightsIH=null;
        double[] weightsHO=null;
        static int numInputs=0;
        double[] trainOutput=null;
        double LR=0.07;
        int numPatterns=0;
        double[] row_ERROR=null;
        double[][] Wi_Hid=null;
        DenseMatrix Wi_Hidden=null ;
        double RMSerror=0;
        //the outputs of the hidden neurons
        public static double[] hiddenVal  =null;
        // this output must be set at some point before close,
        // if there is going to be any output.
        protected OutputCollector<Text,TypedBytesWritable> output;

        public TSQRIteration() {
            this.currentRow = 0;
            this.A = null;
            this.output = null;
            this.rand = new Random();
        }

        public TSQRIteration(int Hidden_Layer,int maxIterations) {
            this();
            this.Hidden_Layer = Hidden_Layer;
            this.maxIterations = maxIterations;
            this.MaxError= MaxError;
            this.LearningRate= LearningRate;
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

        public static double tanh(double x)
        {
            if (x > 20)
                return 1;
            else if (x < -20)
                return -1;
            else
            {
                double a = Math.exp(x);
                double b = Math.exp(-x);
                return (a-b)/(a+b);
            }
        }
         void calcNet()

        {
            hiddenVal = new double[Hidden_Layer];
            //calculate the outputs of the hidden neurons
            //the hidden neurons are tanh

            for(int i = 0;i<Hidden_Layer;i++)
            {
                hiddenVal[i] = 0.0;

                for(int j = 0;j<numInputs;j++)
                    hiddenVal[i] = hiddenVal[i] + (trainInputs[patNum][j] * weightsIH[j][i]);

                hiddenVal[i] = tanh(hiddenVal[i]);
            }

            //calculate the output of the network
            //the output neuron is linear
            outPred = 0.0;

            for(int i = 0;i<Hidden_Layer;i++)
                outPred = outPred + hiddenVal[i] * weightsHO[i];

            //calculate the error
            errThisPat = outPred - trainOutput[patNum];

        }
         double [] WeightChangesHO()
        //adjust the weights hidden-output
        {
            for(int k = 0;k<Hidden_Layer;k++)
            {
                double weightChange = LR * errThisPat * hiddenVal[k];
                weightsHO[k] = weightsHO[k] - weightChange;

                //regularisation on the output weights
                if (weightsHO[k] < -5)
                    weightsHO[k] = -5;
                else if (weightsHO[k] > 5)
                    weightsHO[k] = 5;

            }
            return (weightsHO) ;
        }
        double [][] WeightChangesIH()
        //adjust the weights input-hidden
        {
            for(int i = 0;i<Hidden_Layer;i++)
            {
                for(int k = 0;k<numInputs;k++)
                {
                    double x = 1 - (hiddenVal[i] * hiddenVal[i]);
                    x = x * weightsHO[i] * errThisPat * LR;
                    x = x * trainInputs[patNum][k];
                    double weightChange = x;
                    weightsIH[k][i] = weightsIH[k][i] - weightChange;
                }
            }
            return (weightsIH) ;
        }
         void initWeights()
        {
            weightsHO = new double[Hidden_Layer];
            weightsIH = new double[numInputs][Hidden_Layer];
            for(int j = 0;j<Hidden_Layer;j++)
            {
                weightsHO[j] = (Math.random() - 0.5)/2;
                for(int i = 0;i<numInputs;i++)
                    weightsIH[i][j] = (Math.random() - 0.5)/5;
            }

        }
         void calcOverallError()
        {
            RMSerror = 0.0;
            for(int i = 0;i<numPatterns;i++)
            {
                patNum = i;
                calcNet();
                RMSerror = RMSerror + (errThisPat * errThisPat);
            }
            RMSerror = RMSerror/numPatterns;
            RMSerror = java.lang.Math.sqrt(RMSerror);
        }
        public void collect2(TypedBytesWritable key, TypedBytesWritable value)
                throws IOException
        {

            double row[] = decodeTypedBytesArray(value);

            numInputs=row.length-1;
            row_ERROR=new double [1];
            Wi_output=new double [Hidden_Layer];
            Wi_Hid=new double [row.length-1][Hidden_Layer];

            trainOutput=new double [1];
            trainInputs=new double[1][row.length-1];


            // nombre de lignes
           numPatterns = 1;

            for (int i =0;i<trainInputs.length;i++)
            {
                trainInputs[0][i]=row[i+1];
            }

                trainOutput[0]=row[0];

                //select a pattern at random
                patNum = (int)((Math.random()*numPatterns)-0.001);

                //calculate the current network output
                //and error for this pattern
               calcNet();

                //Recuperer les  network weights  output and Hidden
               Wi_output=WeightChangesHO();
                Wi_Hid=WeightChangesIH();

            Wi_Hidden=new DenseMatrix(Wi_Hid.length,Wi_Hid[0].length) ;

           for(int i = 0; i < Wi_Hidden.numRows(); i++){
                for(int j = 0; j < Wi_Hidden.numColumns(); j++){
                    Wi_Hidden.set(i,j,Wi_Hid[i][j]);
                }
            }

            //display the overall network error
            //after each epoch
            calcOverallError();
            row_ERROR[0]=RMSerror;

        //envoyer l'erreur générée
        Text KeyError = new Text();
        KeyError.set("R_rror".toString());
        output.collect(KeyError, encodeTypedBytes(row_ERROR));

            //envoyer les poids wi_output
            Text KeyWi_Output = new Text();
            KeyWi_Output.set("Out_Wi".toString());
            output.collect(KeyWi_Output, encodeTypedBytes(Wi_output));

            //envoyer les poids wi_Hidden
            Text KeyWi_Hidden = new Text();
            KeyWi_Hidden.set("Hidden_Wi".toString());
            output.collect(KeyWi_Hidden, encodeTypedBytesMatrix(Wi_Hidden));

        }


        public void close() throws IOException
        {

    }
    }
    public static class TSQRMapper
            extends TSQRIteration
            implements Mapper<TypedBytesWritable, TypedBytesWritable,Text, TypedBytesWritable> {

        public void configure(JobConf job){
            this.Hidden_Layer = Integer.parseInt(job.get("Hidden_Layer"));
            this.maxIterations = Integer.parseInt(job.get("maxIterations"));
           //this.MaxError= MaxError;
            //this.LearningRate= Integer.parseInt(job.get("LearningRate"));

        }

        public void map(TypedBytesWritable key, TypedBytesWritable value,
                        OutputCollector<Text,TypedBytesWritable> output,
                        Reporter reporter)
                throws IOException {

            if (this.output == null) {
                this.output = output;
            }
            flag = true;
            numInputs=decodeTypedBytesArray(value).length-1;
            initWeights();
            collect2(key, value);

        }
    }



    public static class TSQRReducer
            extends TSQRIteration
            implements Reducer<Text, TypedBytesWritable,Text, TypedBytesWritable> {
        public void configure(JobConf job){
            this.Hidden_Layer = Integer.parseInt(job.get("Hidden_Layer"));
            this.maxIterations = maxIterations;
            this.MaxError= MaxError;
            //this.LearningRate= Inte.parseInt(job.get("LearningRate"));

        }

        public void reduce(Text key, Iterator<TypedBytesWritable> values,
                           OutputCollector<Text,TypedBytesWritable> output,
                           Reporter reporter)
                throws IOException {
           int sumWi=0;
            double[]  valuesMatrix = null;
            double[] ValuesError=new double[0];
            row_ERROR=new double[1];
            Wi_output=new double [Hidden_Layer];
             double resultWi= 0,resultError=0;
            if (this.output == null) {
                this.output = output;
            }



            while (values.hasNext()) {

                if (key.toString().startsWith("O"))
                {

                    valuesMatrix = decodeTypedBytesArray(values.next());

                        for( int i=0;i<valuesMatrix.length;i++)
                        Wi_output[i]= Wi_output[i]+valuesMatrix[i];

                    sumWi++;

                }


                else if(key.toString().startsWith("R"))
                {   sumWi++;
                    ValuesError = decodeTypedBytesArray(values.next());
                    row_ERROR[0]=row_ERROR[0]+ ValuesError[0];

                }
                else
                {
                    output.collect(key, values.next());
                }
            }
            if (key.toString().startsWith("O"))
            {
                for( int i=0;i<Wi_output.length;i++)
                    Wi_output[i]=Wi_output[i]/sumWi;

                output.collect(key, encodeTypedBytes(Wi_output));
            }
            else if(key.toString().startsWith("R"))
            {
                row_ERROR[0]= row_ERROR[0]/sumWi;
                output.collect(key, encodeTypedBytes(row_ERROR));
            }


        }
       }

    }






