

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
import org.apache.hadoop.mapred.lib.MultipleInputs;
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
import Jama.*;
import weka.core.Instances;

/**
 * Created with IntelliJ IDEA.
 * User: hubiquitus
 * Date: 9/19/14
 * Time: 9:23 AM
 * To change this template use File | Settings | File Templates.
 */





public class  Regression_Validation_parametres extends Configured implements Tool {
    private static final Logger sLogger = Logger.getLogger(Regression_Validation_parametres .class);



    public static void main(String args[]) throws Exception {
        // Let ToolRunner handle generic command-line options
        int res = ToolRunner.run(new Configuration(), new Regression_Validation_parametres (), args);

        System.exit(res);
    }

    private static int printUsage() {
        System.out.println("usage: -mat <filepath> [-output <outputpath>]\n" +
                "  [-nbLignes <int>] [-nbColonnes <int>] ");
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
            outputfile = base + "-Error_param." + ext;
        }

        String nbLignes = getArgument("-nbLignes",args);
        if (nbLignes == null) {
            System.out.println("Required argument '-nbLignes' missing");
        }

        String nbColonnes = getArgument("-nbColonnes",args);
        if (nbColonnes == null) {
            System.out.println("Required argument '-nbColonnes' missing");
        }



        String reduceSchedule = getArgument("-reduce_schedule",args);
        if (reduceSchedule == null) {
            reduceSchedule = "1";
        }



        sLogger.info("Tool name: Regression_Validation_parametres");
        sLogger.info(" -mat: " + matfile);
        sLogger.info(" -output: " + outputfile);
        sLogger.info(" -nbLignes: " +nbLignes );
        sLogger.info(" -nbColonnes: " + nbColonnes);


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
            JobConf conf = new JobConf(getConf(), Regression_Validation_parametres.class);
            DistributedCache.createSymlink(conf);
            conf.setJobName(
                    "Regression_Validation_parametres.java (" + (stage+1) + "/" + stages.length + ")");

            conf.setNumReduceTasks(numReducers);
            conf.setInt("nbLignes", Integer.parseInt(nbLignes));
            conf.setInt("nbColonnes", Integer.parseInt(nbColonnes));
            conf.setInt("stage", stage);
            //onf.set("mapred.child.java.opts","-Xmx1G");

            //conf.addResource(new File("/mnt/var/lib/hadoop/steps/Conf_Regression.xml").toURI().toURL());
            conf.addResource(new File("Conf_Regression.xml").toURI().toURL());

            //get Beta
            int nbCol = Integer.parseInt(nbColonnes);
            conf.setInt("stage", stage);
            String Beta = "";
            for (int s=0;s<nbCol;s++)
            {
                if (s==0)
                    Beta = conf.get("B" + String.valueOf(s));
                else
                    Beta = Beta + "," + conf.get("B" + String.valueOf(s));
            }
            conf.setStrings("Beta",Beta);

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
        static int nbLignes;
        static int nbColonnes;
        String  BetaStrings;
        protected Double numColumns;
        protected int currentRow;
        int row_id = 0;
        List<Integer> row_ids;
        protected Random rand;
        DenseMatrix A=null;
        double SSE=0;
        double SST=0;
        double SSR=0;
        boolean isFirstIteration = false;
        int numReduce = 0;
        int i1=0,j1=0;
        int df = 0;
        double rss = 0.0;      // residual sum of squares
        double ssr = 0.0;
        static Matrix Beta= null;
        // this output must be set at some point before close,
        // if there is going to be any output.
        protected OutputCollector<Text,TypedBytesWritable> output;

        public TSQRIteration() {
            this.currentRow = 0;
            this.A = null;
            this.output = null;
            this.rand = new Random();
        }

        public TSQRIteration(int nbColonnes,int nbLignes) {
            this();
            this.nbLignes = nbLignes;
            this.nbColonnes = nbColonnes;

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
        public void CopierMatrix(Matrix A, Matrix B)
        {
            for (int j=0 ; j<A.getRowDimension(); j++) {
                for (int i=0; i<A.getColumnDimension(); i++) {
                    B.set(j,i,A.get(j,i));
                }
            }
        }
        public void CopierVector(double[] A, double[] B)
        {
            for (int j=0 ; j<A.length; j++) {

                B[j]=A[j];
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


        public void collect2(TypedBytesWritable key, TypedBytesWritable value)
                throws IOException
        {
            double fit=0;
            double[]residual_param=null;
            double row[] = decodeTypedBytesArray(value);
            int nb=nbColonnes;
            int nbY=row.length;
            Matrix X=new Matrix(1,nb);
            Matrix Y=new Matrix(1,1);

            Matrix residuals=new Matrix (1,1);

            for (int i=0;i<nb;i++)
                X.set(0,i,row[i]);

            Y.set(0,0,row[nbY-1]);

            //residuals = (X.times(Beta).minus(Y)).(X.times(Beta).minus(Y));
              fit= (X.times(Beta).minus(Y)).get(0,0);
               fit=fit*fit;


            residual_param=new double[1] ;
            //residual_param[0]=residuals.get(0,0) ;
            residual_param[0]=fit;

            double[] y_param=new double[1] ;
            y_param[0]=Y.get(0,0) ;

            Text residualKey =new Text();
            residualKey.set("Residual".toString());
            Text YKey =new Text();
            YKey.set("Y".toString());

            output.collect(residualKey, encodeTypedBytes(residual_param));
            output.collect(YKey, encodeTypedBytes(y_param));
        }


        public void close() throws IOException
        {

        }
    }
    public static class TSQRMapper
            extends TSQRIteration
            implements Mapper<TypedBytesWritable, TypedBytesWritable,Text, TypedBytesWritable> {

        public void configure(JobConf job){
            this.nbLignes = Integer.parseInt(job.get("nbLignes"));
            this.nbColonnes = Integer.parseInt(job.get("nbColonnes"));
            this.BetaStrings = job.get("Beta");

            String[] BetaString = BetaStrings.split(",");
            Beta = new Matrix(nbColonnes,1);
            for (int s=0;s<nbColonnes;s++)
            {
                Beta.set(s,0,Double.parseDouble(BetaString[s]));
            }

        }

        public void map(TypedBytesWritable key, TypedBytesWritable value,
                        OutputCollector<Text,TypedBytesWritable> output,
                        Reporter reporter)
                throws IOException {

            if (this.output == null) {
                this.output = output;
            }


            collect2(key, value);

        }
    }



    public static class TSQRReducer
            extends TSQRIteration
            implements Reducer<Text, TypedBytesWritable,Text, TypedBytesWritable> {
        public void configure(JobConf job){
            this.nbLignes = Integer.parseInt(job.get("nbLignes"));
            this.nbColonnes = Integer.parseInt(job.get("nbColonnes"));
        }

        public void reduce(Text key, Iterator<TypedBytesWritable> values,
                           OutputCollector<Text,TypedBytesWritable> output,
                           Reporter reporter)
                throws IOException {
            int sumY=0, len =0 ;

            boolean flagR=false;
            boolean flagY=false;
            boolean flag1=false;
            boolean flag2=false;
            double[] valuesMatrix = null;
            double[] ValuesYsum=new double[0];
            double[] ValuesY=new double [1];
            double[] Ytransit=null;
            Matrix ResidualsTotal=null; ;
            double[] Y=new double [1];
            Y[0]=0;
            double R2=0;
            if (this.output == null) {
                this.output = output;
            }



            while (values.hasNext()) {


                if (key.toString().startsWith("R"))
                {

                    valuesMatrix = decodeTypedBytesArray(values.next());
                     SSE=SSE+ valuesMatrix[0];
                     /*
                    if (!flagR){

                        ResidualsTotal = new Matrix(1,1);
                        ResidualsTotal.set(0,0,valuesMatrix[0]);
                        flagR=true;
                    }
                    else {

                        int p= ResidualsTotal.getRowDimension();
                        Matrix ResidualsTemp=new Matrix(p,1) ;
                        CopierMatrix(ResidualsTotal,ResidualsTemp);
                        ResidualsTotal=new Matrix(p+1,1) ;
                        CopierMatrix(ResidualsTemp, ResidualsTotal);

                        ResidualsTotal.set(p,0,valuesMatrix[0]);


                    }
                   */
                }

            else

                if(key.toString().startsWith("Y"))
                {   sumY++;
                    ValuesYsum = decodeTypedBytesArray(values.next());
                    Y[0]=Y[0]+ ValuesYsum[0];
                    if(!flagY)
                    {
                        ValuesY[0]=ValuesYsum[0] ;
                        flagY=true;
                    }
                    else
                    {
                        len=ValuesY.length;
                    Ytransit=new double[len];
                    CopierVector(ValuesY,Ytransit);
                        ValuesY=new double[len+1] ;
                    CopierVector(Ytransit,ValuesY);
                        ValuesY[len]=ValuesYsum[0];
                    }

                }

            }


            if (key.toString().startsWith("R"))
            {
                /*
            }
                flag1=true;
                for (int i=0;i<ResidualsTotal.getRowDimension();i++)
                {

                    SSE= SSE+ ResidualsTotal.get(i,0) ;
                }

               // SSE = ResidualsTotal.norm2() * ResidualsTotal.norm2();

            if (flagY && flagR)
            {
                for (int i=0;i<ResidualsTotal.getRowDimension();i++)
                {

                    SSE= ((ResidualsTotal.get(i,0)-ValuesY[i])*(ResidualsTotal.get(i,0)-ValuesY[i])) ;
                }
              */
                Text keySSE =new Text();
                keySSE.set("keySSE".toString()) ;
                double[] SSE_param=new double[1] ;
                SSE_param[0]=SSE;
                output.collect(keySSE,encodeTypedBytes(SSE_param));

            }

            if (key.toString().startsWith("Y"))
            {
                flag2=true;
                double mean = Y[0] / sumY;
                // total variation to be accounted for
                for (int i = 0; i < sumY; i++) {
                    double dev = ValuesY[i] - mean;
                    SST += dev*dev;
                }
                Text keySST =new Text();
                keySST.set("keySST".toString()) ;
                double[] SST_param=new double[1] ;
                SST_param[0]=SST;
                output.collect(keySST, encodeTypedBytes(SST_param));
            }

            /*
            double dif= SSE/SST;
            R2=1- dif;
            Text keyR2 =new Text();
            keyR2.set("keyR2".toString()) ;
            double[] R2_param=new double[1] ;
            R2_param[0]=R2;
            output.collect(keyR2, encodeTypedBytes(R2_param));

            SSR= SST-SSE;
            Text SSRKey =new Text();
            SSRKey.set("SSR".toString()) ;
            double[] SSR_param=new double[1] ;
            SSR_param [0]=SSR;
            output.collect(SSRKey, encodeTypedBytes(SSR_param));
           */
        }

    }

}






