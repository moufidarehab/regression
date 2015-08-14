/**
 * A native Java implementation of the Tall-and-skinny QR factorization
 * @author David F. Gleich
 */

package gov.sandia.dfgleic;
import Jama.Matrix;
import java.io.*;
import java.util.Random;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.concurrent.Semaphore;
//import java.util.logging.Logger;
import  com.google.common.collect.*;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.QR;
import no.uib.cipr.matrix.*;

import org.apache.hadoop.io.Text;
import java.util.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.mapred.lib.MultipleInputs;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.filecache.DistributedCache;

import org.apache.hadoop.mapred.lib.IdentityMapper;

import org.apache.hadoop.typedbytes.TypedBytesWritable;
import org.apache.hadoop.typedbytes.TypedBytesInput;
import org.apache.hadoop.typedbytes.TypedBytesOutput;
import org.apache.hadoop.typedbytes.Type;

import org.apache.log4j.Logger;



public class  MLR_MR extends Configured implements Tool {
    private static final Logger sLogger = Logger.getLogger(MLR_MR.class);


    public static void main(String args[]) throws Exception {
        // Let ToolRunner handle generic command-line options
        int res = ToolRunner.run(new Configuration(), new MLR_MR(), args);

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
            outputfile = base + "-qrr_update." + ext;
        }

        String reduceSchedule = getArgument("-reduce_schedule",args);
        if (reduceSchedule == null) {
            reduceSchedule = "1";
        }

        String block_Size = getArgument("-block_size",args);
        if (block_Size == null) {
            block_Size = "6";
        }



        String splitSize = getArgument("-split_size",args);

        sLogger.info("Tool name: MLR_MR");
        sLogger.info(" -mat: " + matfile);
        sLogger.info(" -output: " + outputfile);
        sLogger.info(" -reduce_schedule: " + reduceSchedule);
        sLogger.info(" -block_size: " + block_Size);

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
                curoutput = outputfile + "_iter"+(stage+1)+"/part-00000";
            } else {
                curoutput = outputfile;
            }

            // run the iteration
            // TODO make this a separate function?
            JobConf conf = new JobConf(getConf(), MLR_MR.class);
            DistributedCache.createSymlink(conf);
            conf.setJobName(
                    "MLR_MR.java (" + (stage+1) + "/" + stages.length + ")");

            conf.setNumReduceTasks(numReducers);
            conf.setInt("block_size", Integer.parseInt(block_Size));
            conf.setInt("stage", stage);
            //conf.set("mapred.child.java.opts","-Xmx2G");
            if (splitSize != null) {
                conf.set("mapred.minsplit.size", splitSize);
                conf.set("mapreduce.input.fileinputformat.split.minsize", splitSize);
            }

            // set the formats
            conf.setInputFormat(SequenceFileInputFormat.class);
            conf.setOutputFormat(SequenceFileOutputFormat.class);

            // set the data types
            conf.setOutputKeyClass(Text.class);
            conf.setOutputValueClass(TypedBytesWritable.class);

            if (stage ==1) {
                FileInputFormat.setInputPaths(conf, new Path(curinput));

                conf.setMapperClass(IdentityMapper.class);
                conf.setReducerClass(TSQRReducer2.class);
            }
            else if (stage==2)
            {   FileInputFormat.setInputPaths(conf, new Path(curinput));

                conf.setMapperClass(IdentityMapper.class);
                conf.setReducerClass(TSQRReducer3.class);
            } else {
                Path outputPath = new Path(args[3]);

                FileInputFormat.setInputPaths(conf, new Path(curinput));

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
        protected int blockSize;
        protected int numColumns;
        protected int currentRow;
        int row_id = 0;
        List<Integer> row_ids;
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
        double[]Y_init=null;
        double[]Yi=null;
        double[]Ytransit=null;
        boolean isFirstIteration = false;
        int numReduce = 0;
        int i1=0,j1=0;
        ArrayListMultimap<Text, DenseMatrix> Malist = ArrayListMultimap.create();
        // this output must be set at some point before close,
        // if there is going to be any output.
        protected OutputCollector<Text,TypedBytesWritable> output;

        public TSQRIteration() {
            this.numColumns = 0;
            this.blockSize = 6;
            this.currentRow = 0;
            this.A = null;
            this.output = null;
            this.rand = new Random();
        }

        public TSQRIteration(int blockSize) {
            this();
            this.blockSize = blockSize;
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
        public boolean verifyZero(double[] row)
        {        boolean Notvide=false;

            for (int j=0;j<row.length;j++)
            {
                if(row[j]!=0)
                    Notvide =true;
            }
            return Notvide;
        }

        public void CopierMatrix(DenseMatrix A, DenseMatrix B)
        {
            for (int j=0 ; j<A.numRows(); ++j) {
                for (int i=0; i<A.numColumns(); ++i) {
                    B.set(j,i,A.get(j,i));
                }
            }
        }

        public void CopierVector(double[] A, double[] B)
        {
            for (int j=0 ; j<A.length; ++j) {

                B[j]=A[j];
            }
        }
        public void CopierUpperTriangDenseMatrix(UpperTriangDenseMatrix A, DenseMatrix B)
        {

            for (int j=0 ; j<A.numRows(); ++j) {
                for (int i=0; i<A.numColumns(); ++i) {
                    B.set(j,i,A.get(j,i));
                }
            }
        }

        public DenseMatrix Clean_matrix (DenseMatrix Q)
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
        public double[] Clean_array (double [] Q)
        {

            double[] result = new double[Q.length];

            for (int j=0;j<result.length;j++)
            {


                result[j]=Q[j];
            }
            return result;
        }

        public void compress() throws IOException
        {

            A1=new DenseMatrix(Y_A.length,1);
            for (int p=0;p<Y_A.length;p++)
            {
                A1.set(p,0,Y_A[p]) ;
            }
            Text key =new Text();
            //envoi du y
            key.clear();
            String Name="Q".concat(String.valueOf(row_id));
            key.set(Name.toString());
            output.collect(key, encodeTypedBytesMatrix(Clean_matrix(A1)));

            QR qr = QR.factorize(A);
            UpperTriangDenseMatrix Ri = qr.getR();
            DenseMatrix Qi = Clean_matrix(qr.getQ());

            //Construire les clés Qi et les envoyer
            key.clear();
            Name="Q".concat(String.valueOf(row_id));
            key.set(Name.toString());
            output.collect(key, encodeTypedBytesMatrix(Qi));

            //Construire les clés Ri et les envoyer

            R_int=new DenseMatrix(Ri.numRows(),Ri.numColumns()) ;
            CopierUpperTriangDenseMatrix(Ri,R_int);
            key.clear();
            Name="R".concat(String.valueOf(row_id));
            key.set(Name.toString());
            output.collect(key, encodeTypedBytesMatrix(R_int));

            A.zero();
            A=null;
            Y_A=null;
            Y_A=null;

        }



        public void collect2(TypedBytesWritable key, TypedBytesWritable value)
                throws IOException
        {

            double row[] = decodeTypedBytesArray(value);

            if (currentRow == 0)
                row_id = (Integer) key.getValue();


            if (A == null)
            {
                numColumns = row.length-1;
                A = new DenseMatrix(numColumns*blockSize,numColumns);
                Y_A = new double[numColumns*blockSize];
            }

            for (int i=0; i<row.length; ++i) {
                if (i == row.length-1)
                    Y_A[currentRow]= row[i];
                else
                    A.set(currentRow, i, row[i]);
            }
            currentRow ++;

            if (currentRow >= A.numRows()) {
                compress();
                currentRow=0;
            }
        }

        public void close() throws IOException
        {  Text key =new Text();
            if (output != null) {

                if(flag==true)  //MAP1
                {
                    if (A != null)
                    {
                        compress();
                        //flag = false;
                    }
                }

                else if (numReduce==1) //reduce1 firstIteration
                {    int row_num=0;
                    int bloc=0;
                    double[] y=null;

                    //factorize R_temp

                    A=new DenseMatrix(R_int.numRows(),R_int.numColumns());
                    CopierMatrix(R_int,A);
                    QR qr = QR.factorize(A);
                    UpperTriangDenseMatrix Ri = qr.getR();
                    DenseMatrix Qi = qr.getQ();

                    //send R final

                    Text KeyR = new Text();
                    KeyR.set("R".toString());
                    R_int = new DenseMatrix(Ri.numRows(),Ri.numColumns());
                    CopierUpperTriangDenseMatrix(Ri,R_int);
                    output.collect(KeyR,encodeTypedBytesMatrix(R_int));

                    //Decomposer Qi en petits bloc Qi'

                    double array[] = new double[Qi.numColumns()];
                    Q3=new DenseMatrix(Qi.numColumns(),Qi.numColumns());
                    int numBloc = 0;
                    for (int i=0;i<Qi.numRows();i++) {
                        for (int j=0; j<Qi.numColumns(); ++j) {
                            array[j] = Qi.get(i,j);
                            Q3.set(i1,j1,array[j]);
                            j1++;
                        }
                        i1++;j1=0;
                        if(i1==Q3.numRows())

                        {
                            //envoyer les Qi avec yi

                            key.clear();
                            String Name="Q".concat(String.valueOf(row_ids.get(numBloc)));
                            key.set(Name.toString());
                            output.collect(key, encodeTypedBytesMatrix(Q3));

                            numBloc++;
                            i1=0;j1=0;

                        }
                    }


                }

                else if(numReduce==2)  //reduce2 secondIteration
                {
                    Text KeyFinal = new Text();
                    KeyFinal.set("final".toString());

                    output.collect(KeyFinal, encodeTypedBytesMatrix(A1));

                }
                else if(numReduce==3)  //reduce3 ThirdIteration
                {
                    DenseVector B=new DenseVector(R.numColumns());

                    //METTRE R dans RI DE TYPE MATRIX POUR L'inverser

                    Matrix RI=new Matrix(R.numRows(),R.numColumns()) ;
                    for (int j=0 ; j<RI.getRowDimension(); j++) {
                        for (int i=0; i<RI.getColumnDimension(); i++) {
                            RI.set(j,i,R.get(j,i));
                        }
                    }
                    RI=RI.inverse();

                    //le remettre dans R

                    R = new DenseMatrix(RI.getRowDimension(),RI.getColumnDimension());
                    for (int i=0;i<R.numRows();i++)
                    {
                        for (int j=0;j<R.numColumns();j++)
                        {
                            R.set(i,j,RI.get(i,j));
                        }
                    }

                    R.mult(Val, B);

                    key.set("B".toString());
                    output.collect(key, encodeTypedBytes(B.getData()));

                }
            }
        }
    }
    public static class TSQRMapper
            extends TSQRIteration
            implements Mapper<TypedBytesWritable, TypedBytesWritable,Text, TypedBytesWritable> {

        public void configure(JobConf job){
            this.blockSize = Integer.parseInt(job.get("block_size"));
            this.isFirstIteration = (Integer.parseInt(job.get("stage")) == 0);

        }

        public void map(TypedBytesWritable key, TypedBytesWritable value,
                        OutputCollector<Text,TypedBytesWritable> output,
                        Reporter reporter)
                throws IOException {

            if (this.output == null) {
                this.output = output;
            }
            flag = true;

            collect2(key, value);

        }
    }



    public static class TSQRReducer
            extends TSQRIteration
            implements Reducer<Text, TypedBytesWritable,Text, TypedBytesWritable> {
        public void configure(JobConf job){
            this.blockSize = Integer.parseInt(job.get("block_size"));
            this.isFirstIteration = (Integer.parseInt(job.get("stage")) == 0);

        }

        public void reduce(Text key, Iterator<TypedBytesWritable> values,
                           OutputCollector<Text,TypedBytesWritable> output,
                           Reporter reporter)
                throws IOException {
            // Q1.zero();
            if (this.output == null) {
                this.output = output;
            }
            currentRow=0;
            numReduce = 1;

            DenseMatrix  valuesMatrix = null;



            while (values.hasNext()) {

                if (key.toString().startsWith("Q"))
                {
                    output.collect(key, values.next());
                   Malist.put(key,decodeTypedBytesMatrix(values.next())) ;
                }
                else
                {

                    valuesMatrix = decodeTypedBytesMatrix(values.next());
                    //output.collect(key,encodeTypedBytesMatrix(valuesMatrix));
                    if (row_ids == null)
                    {
                        row_ids = new ArrayList<Integer>();
                    }
                    row_ids.add(Integer.parseInt(key.toString().substring(1)));

                    if(NVide== false)
                    {   NVide=true;
                        R_int=new DenseMatrix(valuesMatrix.numRows(),valuesMatrix.numColumns()) ;
                        CopierMatrix(valuesMatrix,R_int);
                    }
                    else
                    {
                        R=new DenseMatrix(R_int.numRows(),R_int.numColumns()) ;
                        CopierMatrix(R_int,R);
                        R_int=new DenseMatrix(R.numRows()+ valuesMatrix.numRows(),R.numColumns()) ;
                        CopierMatrix(R,R_int);

                        for (int j=R.numRows(), j1=0 ; j1<valuesMatrix.numRows(); j++,++j1) {
                            for (int i=0; i<valuesMatrix.numColumns(); ++i) {
                                R_int.set(j, i, valuesMatrix.get(j1, i));
                            }
                        }
                    }

                }
            }

        }
    }

    public static class TSQRReducer2
            extends TSQRIteration
            implements Reducer<Text, TypedBytesWritable,Text, TypedBytesWritable> {
        public void configure(JobConf job){
            this.blockSize = Integer.parseInt(job.get("block_size"));
            this.isFirstIteration = (Integer.parseInt(job.get("stage")) == 0);

        }

        public void reduce(Text key, Iterator<TypedBytesWritable> values,
                           OutputCollector<Text,TypedBytesWritable> output,
                           Reporter reporter)
                throws IOException {
            // Q1.zero();
            if (this.output == null) {
                this.output = output;
            }
            numReduce = 2;
            boolean flagQ1=false;
            boolean flagQ2=false;
            DenseVector Y=null;
            double [] y=null;

            while (values.hasNext()) {

                // recupere le R

                if (key.toString().equals("R"))
                {
                    Text KeyFinal = new Text();
                    KeyFinal.set("R".toString());

                    output.collect(KeyFinal, values.next());
                }

                //Recuperer les Qi et Qi' et yi car ils ont la meme clÃ©
                else
                {
                    A = Clean_matrix(decodeTypedBytesMatrix(values.next()));

                    //detecter si values est le yi

                    if (A.numColumns()==1)
                    {
                        y=new double[A.numRows()];

                        for (int i=0;i<A.numRows();i++)
                        {
                            y[i]=A.get(i,0);
                        }
                    }
                    // sinon les Qi

                    else
                    {
                        if (!flagQ1)
                        {
                            //Q1 = Clean_matrix(decodeTypedBytesMatrix(values.next()));
                            Q1=new DenseMatrix(A.numRows(),A.numColumns());
                            CopierMatrix(A,Q1);
                            flagQ1 = true;
                        }
                        else if (!flagQ2)
                        {
                            //Q2 = Clean_matrix(decodeTypedBytesMatrix(values.next()));
                            Q2=new DenseMatrix(A.numRows(),A.numColumns());
                            CopierMatrix(A,Q2);
                            flagQ2 = true;
                        }

                    }

                }

            }
            //ordonner les Qi

            if (flagQ1 && flagQ2 )
            {
                if (Q1.numColumns()==Q2.numRows())
                {
                    Q3=new DenseMatrix(Q1.numRows(),Q2.numColumns());
                    Q1.mult(Q2, Q3);
                }

                else if (Q2.numColumns()==Q1.numRows())
                {
                    Q3=new DenseMatrix(Q2.numRows(),Q1.numColumns());
                    Q2.mult(Q1, Q3);

                }

                //METTRE Q3 dans Q12 DE TYPE MATRIX POUR LA TRANSPOSER

                Matrix Q12=new Matrix(Q3.numRows(),Q3.numColumns()) ;
                for (int j=0 ; j<Q12.getRowDimension(); j++) {
                    for (int i=0; i<Q12.getColumnDimension(); i++) {
                        Q12.set(j,i,Q3.get(j,i));
                    }
                }
                //Transpose

                Q12=Q12.transpose();

                // REMETTRE DANS Q1 DE TYPE DENSMATRIX

                Q1 = new DenseMatrix(Q12.getRowDimension(),Q12.getColumnDimension());
                for (int i=0;i<Q1.numRows();i++)
                {
                    for (int j=0;j<Q1.numColumns();j++)
                    {
                        Q1.set(i,j,Q12.get(i,j));
                    }
                }

                Y = new DenseVector(y.length);

                //Remplir Y depuis y

                for (int h=0; h<Y.size(); ++h) {
                    Y.set(h,y[h]);
                }

                DenseVector Y1 = new DenseVector(Q1.numRows());

                //QT*y

                Q1.mult(Y, Y1);


                //recuperer les Vi[]

                if(!flagY)
                { flagY=true;
                    A1=new DenseMatrix(1,Y1.size());
                    for(int i=0;i<Y1.size();i++)
                        A1.set(0,i,Y1.get(i));
                }
                else
                {
                    A=new DenseMatrix(A1.numRows(),A1.numColumns());
                    CopierMatrix(A1,A);
                    A1=new DenseMatrix(A.numRows()+1,A.numColumns());
                    CopierMatrix(A,A1);

                    for(int i=0;i<Y1.size();i++)
                        A1.set(A.numRows(),i,Y1.get(i));

                }

            }


        }

    }



    public static class TSQRReducer3
            extends TSQRIteration
            implements Reducer<Text, TypedBytesWritable,Text, TypedBytesWritable> {
        public void configure(JobConf job){
            this.blockSize = Integer.parseInt(job.get("block_size"));
            this.isFirstIteration = (Integer.parseInt(job.get("stage")) == 0);

        }

        public void reduce(Text key, Iterator<TypedBytesWritable> values,
                           OutputCollector<Text,TypedBytesWritable> output,
                           Reporter reporter)
                throws IOException {

            if (this.output == null) {
                this.output = output;
            }
            numReduce = 3;

            while (values.hasNext()) {
                if (key.toString().equals("R"))
                {
                    R = decodeTypedBytesMatrix(values.next());
                }
                else
                {

                    A1 = decodeTypedBytesMatrix(values.next());
                    Val=new DenseVector(A1.numColumns());
                    for (int j=0;j<A1.numColumns();j++)
                    {
                        double sum = 0;
                        for(int i=0;i<A1.numRows();i++)
                        {
                            sum = sum + A1.get(i,j);
                        }

                        Val.set(j,sum);
                    }

                }

            }


        }

    }
}




