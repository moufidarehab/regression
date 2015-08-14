package gov.sandia.dfgleic;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
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
import weka.core.*;

import java.io.*;
import java.util.*;


/**
 * Created with IntelliJ IDEA.
 * User: hubiquitus
 * Date: 3/27/15
 * Time: 4:02 PM
 * To change this template use File | Settings | File Templates.
 */
public class Bagging_Selection_Map_Backward extends Configured implements Tool {
    private static final Logger sLogger = Logger.getLogger(Bagging_Selection_Map_Backward.class);



    public static void main(String args[]) throws Exception {
        // Let ToolRunner handle generic command-line options
        int res = ToolRunner.run(new Configuration(), new Bagging_Selection_Map_Backward(), args);

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

        String PathPruning = getArgument("-prun_file",args);
        if (PathPruning == null) {
            System.out.println("Required argument '-prun_file' missing");
            return -1;
        }

        String nbTest = getArgument("-nbTest",args);
        if (nbTest == null) {
            System.out.println("Required argument '-nbTest' missing");
            return -1;
        }

        String nbSelect = getArgument("-nbSelect",args); //facultatif


        String splitSize = getArgument("-split_size",args);

        sLogger.info("Tool name: Bagging_Selection");
        sLogger.info(" -mat: " + matfile);
        sLogger.info(" -output: " + outputfile);
        sLogger.info(" -reduce_schedule: " + reduceSchedule);
        sLogger.info(" -block_size: " + block_Size);
        sLogger.info(" -prun_file: " + PathPruning);
        sLogger.info(" -test_file: " + PathTest);
        sLogger.info(" -Bag_Iter: " + nbBagIter);
        sLogger.info(" -nbTest: " + nbTest);
        sLogger.info(" -nbSelect: " + nbSelect);
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
            JobConf conf = new JobConf(getConf(), Bagging_Selection_Map_Backward.class);
            DistributedCache.createSymlink(conf);
            conf.setJobName(
                    "Bagging_Selection.java (" + (stage+1) + "/" + stages.length + ")");

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
            conf.set("prun_path",PathPruning);
            conf.set("nbTest",nbTest);

            //partage du nb model à selectionner
            if (nbSelect != null)
                conf.set("nbSelect",nbSelect);
            else
                conf.set("nbSelect","-1");
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
                conf.setReducerClass(BaggingReducerTest.class);
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

        protected Hashtable<String,Double> Somm_Residual_ALL;
        protected Hashtable<String,DenseVector> Y_Prob_ALL;
        protected Hashtable<String,DenseVector> H_ALL;
        protected DenseVector Current_Y_Prob;
        protected Double Current_MSE;
        protected DenseVector Somm_Y_ALL;
        String TestPath;
        String PrunPath;
        protected int nbTest;
        protected int nbSelect;
        protected int nbBagIter;
        protected int blockSize;
        protected int numColumns;
        protected int nbColumns;
        protected int currentRow;

        protected int numReduce;
        protected DenseVector Y_reel;
        Double Agregate;
        CSVReader products;
        int nbModel;
        double[] Y_test;
        Long row_id;
        ArrayList<Long> row_ids;
        protected Random rand;
        DenseMatrix A;

        boolean flag=false;
        boolean isFirstIteration = false;

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

        //calcul MSE depuis somme des residual dans vecteur
        public Double ComputeMSE(DenseVector Y_SOMM, boolean apply_SQRT,int nbModel)
        {
            Double Agregate = 0.0 ;

            for (int j=0;j<Y_SOMM.size();j++)
            {
                Agregate = Agregate + (Y_SOMM.get(j)/nbModel - Y_reel.get(j)) * (Y_SOMM.get(j)/nbModel - Y_reel.get(j));
            }
            Double MSE = Agregate/nbTest;
            if (apply_SQRT)
                MSE = Math.sqrt(MSE);

            return MSE;

        }

        //calcul MSE depuis somme des residual dans vecteur
        public Double ComputeMSE(Collection<Double> Y_SOMM, boolean apply_SQRT)
        {
            Double Agregate = 0.0 ;

            Iterator<Double> t = Y_SOMM.iterator();
            while (t.hasNext())
            {
                Agregate = Agregate + t.next();
            }
            int nbDIV = Y_SOMM.size();
            Double MSE = Agregate/(nbDIV*nbTest);  // 1/N*|S| Somm (y-y')2
            if (apply_SQRT)
                MSE = Math.sqrt(MSE);

            return MSE;

        }

        //select le min dans un ensemble de residual (y-y')2
        public String ComputeMIN_MSE(Hashtable<String,Double> Somm_Residual_ALL)
        {
            Double MIN_MSE = Double.MAX_VALUE;
            String key_select = "";
            Iterator<String> k = Somm_Residual_ALL.keySet().iterator();
            while(k.hasNext())
            {
                String key = k.next();
                if (Somm_Residual_ALL.get(key) < MIN_MSE)
                {
                    MIN_MSE = Somm_Residual_ALL.get(key);
                    key_select = key;
                }
            }

            Current_MSE = MIN_MSE/nbTest;
            //retourner la clé en Y_indexBlock_IndexModel (utilisé dans H_S et S) au lieu de la clé residual qui est utilisé juste pour la premiere selection
            int index_code = "Y_Residual_".length();
            return "Y_" + key_select.substring(index_code);
        }

        public DenseVector Sub_DenseVector(DenseVector A, DenseVector B)
        {
            DenseVector C = new DenseVector(A.size());

            for (int i=0;i<A.size();i++)
                C.set(i,A.get(i) - B.get(i));

            return  C;
        }

        public Hashtable<String,DenseVector> Construct_Models_Table (DenseMatrix M, String keyPrefix)
        {
            Hashtable<String,DenseVector> Table = new Hashtable<String, DenseVector>();
            for (int i=0;i<M.numRows();i++)
            {
                DenseVector Model = new DenseVector(M.numColumns());
                for (int j=0;j<M.numColumns();j++)
                {
                  Model.set(j,M.get(i,j));
                }
                Table.put(keyPrefix + "_" + String.valueOf(i), Model);
            }
            return Table;
        }

                //function selection itération    : comparaison avec l'ancien MSE
                public String SelectONE_Backward(Hashtable<String, DenseVector> S, Hashtable<String, DenseVector> H_S) throws IOException
                {
                    String key_select = "";
                    if (S.size() == 0) //cas du début de la selection
                    {
                        Current_Y_Prob = Somm_Y_ALL;
                    }


                    Iterator<String> k = H_S.keySet().iterator();
                    while(k.hasNext())
                    {
                        String key_candidate = k.next();
                        DenseVector candidate_Y_Prob = H_S.get(key_candidate); //estimation du modèle à selectionner
                        DenseVector new_Y_Prob = Sub_DenseVector(Current_Y_Prob,candidate_Y_Prob); //somme des modèles + modèle à selectionner
                        Double new_Residual = 0.0;
                        int S_Size = S.size() + 1; //|S| + 1 pour le modèle à selectionner
                        for (int i=0;i<new_Y_Prob.size();i++)
                        {
                            Double new_Y = (new_Y_Prob.get(i)/S_Size - Y_reel.get(i))*(new_Y_Prob.get(i)/S_Size - Y_reel.get(i)); // (y/|S|-y')2
                            new_Residual = new_Residual + new_Y;

                        }

                        Double candidate_MIN_MSE = new_Residual/nbTest;
                        //Debug
                        double[] MSE_param = new double[1];
                        MSE_param[0] = Math.sqrt(candidate_MIN_MSE);
                        // Affichage des candidats à la selection
                        // output.collect(new Text(key_candidate + "_Size : " + String.valueOf(S.size())),encodeTypedBytes(MSE_param));



                        if (candidate_MIN_MSE < Current_MSE)
                        {
                            Current_MSE = candidate_MIN_MSE;
                            key_select = key_candidate;
                            Current_Y_Prob = new_Y_Prob;
                        }
                    }
                    return key_select;
                }

        //Affichage en mode debug des residus
        public void Affiche_Debug()
        {
            CSVWriter csv = null;
            try {
                csv = new CSVWriter(new FileWriter("/home/hduser/testHash.tmat"));
            } catch (IOException e) {
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }
            Iterator<String> k = Somm_Residual_ALL.keySet().iterator();
            while (k.hasNext())
            {   String cle = k.next();
                String[] s = new String[2];
                s[0] = cle;
                s[1] = Somm_Residual_ALL.get(cle).toString();
                csv.writeNext(s);

            }

            k = Y_Prob_ALL.keySet().iterator();
            while (k.hasNext())
            {   String cle = k.next();
                String[] s = new String[nbTest+1];
                DenseVector t = Y_Prob_ALL.get(cle);
                for (int i=0; i<t.size();i++ )
                    s[i+1] = String.valueOf(t.get(i));
                csv.writeNext(s);

            }
            try {
                csv.close();
            } catch (IOException e) {
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }
        }

        //Backward selection
        public Hashtable<String, DenseVector> Backward_Selection()  throws IOException
        {
            Hashtable<String, DenseVector> S = new Hashtable<String, DenseVector>();
            Hashtable<String, DenseVector> H_S = new Hashtable<String, DenseVector>(Y_Prob_ALL);

            boolean endSelection = false;
            Current_MSE = Double.MAX_VALUE;
            String key = "";
            while (!endSelection) {

                key = SelectONE_Backward(S, H_S); //comparaison avec le MSE de l'itération précédente

                //plus aucune optimisation possible
                if (key.length() == 0) break;
                //add to S and removre from H-S
                S.put(key, H_S.get(key));
                H_S.remove(key);
                //Afficher resultat de la selection
                double[] MSE_param = new double[1];
                MSE_param[0] = Math.sqrt(Current_MSE);
                output.collect(new Text(key + "_Selected_Size : " + String.valueOf(H_S.size())),encodeTypedBytes(MSE_param));
                //case nbselect fixé ou S=H
                if ((nbSelect > 0 && H_S.size() == nbSelect) || (S.size() == 0))
                {
                    endSelection = true;
                }
            }

            return H_S;
        }
        //calcul evaluation pour le MAP
        public void Evaluate_Models(DenseMatrix OutWeight)
        {
            try {
                //recuperation du fichier de test
                this.products = new CSVReader(new FileReader(this.PrunPath));

            } catch (IOException e) {
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }

            //Construire la liste des modèle du bloc en cours
            Hashtable<String, DenseVector> Models = Construct_Models_Table(OutWeight,"H_".concat(String.valueOf(row_id)));

            //initialisation
            String[] nextLine;

            DenseVector Y_Prob_Line = new DenseVector(OutWeight.numRows());
            DenseVector Y_Residual = new DenseVector(OutWeight.numRows());
            Somm_Residual_ALL = new Hashtable<String, Double>();
            Y_Prob_ALL = new Hashtable<String, DenseVector>();
            DenseVector Y_SOMM = new DenseVector(nbTest);

            Current_MSE = Double.MAX_VALUE;

            Y_Residual.zero();
            boolean send_Y_reel = false;
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
                    OutWeight.mult(XTest_V,Y_Prob_Line);
                    Double Somm_Y_Prob = 0.0;
                    for (int i=0;i<Y_Prob_Line.size();i++)
                    {
                        String keyResidual = "Y_Residual_" + String.valueOf(row_id) + "_" + String.valueOf(i);
                        //Construction de Y_Prob_ALL
                        String key = "Y_" + String.valueOf(row_id) + "_" + String.valueOf(i);
                        DenseVector Y_Prob_Model;
                        if (iLine == 0)
                        {
                           Y_Prob_Model = new DenseVector(nbTest);
                           Somm_Residual_ALL.put(keyResidual, ((Y_Prob_Line.get(i) - Y) * (Y_Prob_Line.get(i) - Y)));
                        }
                        else
                        {
                            Y_Prob_Model = Y_Prob_ALL.get(key);
                            Somm_Residual_ALL.put(keyResidual, Somm_Residual_ALL.get(keyResidual) + ((Y_Prob_Line.get(i) - Y) * (Y_Prob_Line.get(i) - Y)));
                        }
                        Somm_Y_Prob = Somm_Y_Prob + Y_Prob_Line.get(i); //(y1+y2/|S|-y')2
                        Y_Prob_Model.set(iLine,Y_Prob_Line.get(i));
                        Y_Prob_ALL.put(key, Y_Prob_Model);

                    }
                    Y_SOMM.set(iLine, Somm_Y_Prob);
                    iLine++;
                }

                //Selection dans le MAP
                //Affichage des résidus en mode debug + Y probable
                //Affiche_Debug();
                //Calcul MSE ALL
                Double MSE_H = ComputeMSE(Y_SOMM,true,OutWeight.numRows()); //appliquer la racine pour valeur finale;
                double[] MSE_param = new double[1];
                MSE_param[0] = MSE_H;
                output.collect(new Text("MSE_ALL_".concat(String.valueOf(row_id))), encodeTypedBytes(MSE_param));

                //selection Backward
                Current_MSE = MSE_H * MSE_H;
                Hashtable<String, DenseVector> S = Backward_Selection();

                //Afficher MSE selected
                //MSE_param[0] = Math.sqrt(Current_MSE);
                //output.collect(new Text("MSE_S_".concat(String.valueOf(row_id).concat("_Size_").concat(String.valueOf(S.size())))), encodeTypedBytes(MSE_param));

                //Afficher modeles selectionnés depuis H_ALL qui contient les beta
                for (String key : S.keySet())
                {
                    //envoyer les probables
                    output.collect(new Text(key), encodeTypedBytes(S.get(key).getData()));
                    //envoyer les modeles
                    String key_H = key.replace('Y', 'H');
                    output.collect(new Text(key_H), encodeTypedBytes(Models.get(key_H).getData()));  //envoyer juste les clés de ce qui est selectionner
                    //envoyer les residual
                    String key_Residual = key.replace("Y_", "Y_Residual_");
                    double[] value = new double[1];
                    value[0] = Somm_Residual_ALL.get(key_Residual);
                    output.collect(new Text(key_Residual), encodeTypedBytes(value));

                }

                //Envoyer Y_Reel une seule fois
                if (send_Y_reel) //Envoyer Y_Reel une seule fois
                {
                    Text KeyY = new Text();
                    KeyY.set("Y_Reel".toString());
                    output.collect(KeyY, encodeTypedBytes(Y_reel.getData()));
                }
                //Envoyer Somm_Y residual du bloc actuel
                Text KeyYSOMM = new Text();
                KeyYSOMM.set("Y_SOMM_".toString().concat(String.valueOf(row_id)));
                output.collect(KeyYSOMM, encodeTypedBytes(Y_SOMM.getData()));

            } catch (IOException e) {
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }

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
                Evaluate_Models(Compress(A)); //Column correspond nb modele = iteration du classifier, ligne correspond au nb de test
                //DenseMatrix M = Compress(A);
                //output.collect(new Text("B"),encodeTypedBytesMatrix(M));
                A = null;
            }
        }
        public void close() throws IOException
        {

            if (output != null)
                if (is_reduce && numReduce == 1) {

                    //Affichage des résidus en mode debug + Y probable
                    //Affiche_Debug();
                    //Calcul MSE ALL
                    Double MSE_ALL = ComputeMSE(Somm_Y_ALL, true, nbModel); //appliquer la racine pour valeur finale;
                    double[] MSE_param = new double[1];
                    MSE_param[0] = MSE_ALL;
                    output.collect(new Text("Final_Distributed_MSE_ALL"), encodeTypedBytes(MSE_param));
                    MSE_param[0] = nbModel;
                    output.collect(new Text("Final_Distributed_NbModels_ALL"),encodeTypedBytes(MSE_param));

                    //selection Backward
                    Current_MSE = MSE_ALL * MSE_ALL;
                    Hashtable<String, DenseVector> S = Backward_Selection();

                    //Afficher MSE selected
                    MSE_param[0] = Math.sqrt(Current_MSE);
                    output.collect(new Text("Final_Distributed_MSE_S"), encodeTypedBytes(MSE_param));
                    MSE_param[0] = S.size();
                    output.collect(new Text("Final_Distributed_NbModels_S"), encodeTypedBytes(MSE_param));
                    //Afficher modeles selectionnés depuis H_ALL qui contient les beta
                    for (String key : S.keySet())
                    {
                        String key_H = key.replace("Y", "H");
                        output.collect(new Text(key_H), encodeTypedBytes(H_ALL.get(key_H).getData()));
                    }


                }
                else if (is_reduce && numReduce ==2)
                {
                    this.products = new CSVReader(new FileReader(this.TestPath));
                    String[] nextLine;
                    Y_reel = new DenseVector(nbTest);
                    int iLine = 0;
                    //construire la matrice OutWeight
                    DenseMatrix OutWeight = new DenseMatrix(H_ALL.size(), nbColumns);
                    int k = 0;
                    for (String key : H_ALL.keySet())
                    {
                        DenseVector H = H_ALL.get(key);
                        for (int j=0;j< nbColumns;j++)
                        {
                            OutWeight.set(k, j, H.get(j));
                        }
                        k++;
                    }

                    nbModel = OutWeight.numRows();
                    //estimation
                    DenseVector Y_Prob_Line = new DenseVector(nbModel);
                    DenseVector Y_SOMM = new DenseVector(nbTest);

                    while ((nextLine = products.readNext()) != null)
                    {
                        DenseVector XTest_V = new DenseVector(nextLine.length);

                        for (int i=0;i<nextLine.length;i++)
                        {
                            if (i==nextLine.length-1)
                            {
                                Double Y = Double.parseDouble(nextLine[i]);
                                XTest_V.set(i,1);

                                Y_reel.add(iLine, Y);
                            }
                            else
                                XTest_V.set(i,Double.parseDouble(nextLine[i]));
                        }
                        OutWeight.mult(XTest_V,Y_Prob_Line);
                        Double Somm_Y_Prob = 0.0;
                        for (int i=0;i<Y_Prob_Line.size();i++)
                        {
                            Somm_Y_Prob = Somm_Y_Prob + Y_Prob_Line.get(i); //(y1+y2/|S|-y')2
                        }
                        Y_SOMM.set(iLine, Somm_Y_Prob);
                        iLine++;
                    }

                    //calcul du MSE
                    Double Agregate = 0.0;
                    for (int i=0;i<nbTest;i++)
                    {
                        Agregate = Agregate + (((Y_SOMM.get(i)/nbModel)-Y_reel.get(i)) * ((Y_SOMM.get(i)/nbModel)-Y_reel.get(i)));

                    }
                    double[] MSE_param = new double[1];
                    MSE_param[0] = Math.sqrt(Agregate/nbTest);
                    output.collect(new Text("MSE_TEST"), encodeTypedBytes(MSE_param));

                }
                else if (A != null) {

                    DenseMatrix A_Cleaned = Clean_matrix(A);
                    try {
                        Evaluate_Models(Compress(A_Cleaned));
                        //DenseMatrix M = Compress(A_Cleaned);
                        //output.collect(new Text("B"),encodeTypedBytesMatrix(M));

                    } catch (Exception e) {
                        e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
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

            this.nbSelect = Integer.parseInt(job.get("nbSelect"));
            this.PrunPath = job.get("prun_path");

            this.Agregate = 0.0;
            Somm_Residual_ALL = new Hashtable<String,Double>(); //pour garder trace de la selection et faire correspondre dans H

            Y_Prob_ALL = new Hashtable<String,DenseVector>();
            H_ALL = new Hashtable<String,DenseVector>();

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
            Somm_Residual_ALL = new Hashtable<String,Double>(); //pour garder trace de la selection et faire correspondre dans H

            Y_Prob_ALL = new Hashtable<String,DenseVector>();
            H_ALL = new Hashtable<String,DenseVector>();
            this.nbTest = Integer.parseInt(job.get("nbTest"));
            this.nbBagIter = Integer.parseInt(job.get("Bag_Iter"));
            Somm_Y_ALL = new DenseVector(nbTest);

        }


        public void reduce(Text key, Iterator<TypedBytesWritable> values,
                           OutputCollector<Text,TypedBytesWritable> output,
                           Reporter reporter)
                throws IOException {
            if (this.output == null) {
                this.output = output;
            }

            is_reduce = true;
            numReduce =1;

            while (values.hasNext()) {
                //output.collect(key,values.next());

                if (key.toString().startsWith("H")) //sortir les modèles
                {
                    double[] Model = decodeTypedBytesArray(values.next());
                    //On affiche pas tout les modèles : on affichera que la selection
                    //output.collect(new Text(newKey),encodeTypedBytes(Model.getData()));
                    H_ALL.put(key.toString(),new DenseVector(Model));

                }
                else if (key.toString().equals("Y_Reel"))
                {
                    Y_reel = new DenseVector(decodeTypedBytesArray(values.next()));
                }
                else if (key.toString().startsWith("Y_Residual_"))  //recuperation de la somme des residuals
                {
                    double[] Y_Residual = decodeTypedBytesArray(values.next());
                    Somm_Residual_ALL.put(key.toString(),Y_Residual[0]);

                }
                else if (key.toString().startsWith("Y_SOMM"))
                {
                    double[] Y_SOMM = decodeTypedBytesArray(values.next());

                    for (int j=0;j<Y_SOMM.length;j++)
                    {
                        Somm_Y_ALL.set(j, Somm_Y_ALL.get(j) + Y_SOMM[j]);
                    }

                    nbModel = nbModel + nbBagIter;
                }
                else if (key.toString().startsWith("Y_"))  //recuperation   des estimations
                {
                    double[] Y_Prob = decodeTypedBytesArray(values.next());
                    Y_Prob_ALL.put(key.toString(),new DenseVector(Y_Prob));
                }
                else
                {
                    output.collect(key,values.next());
                }


            }

        }
    }

    public static class BaggingReducerTest
            extends BaggingIteration
            implements Reducer<Text, TypedBytesWritable,Text, TypedBytesWritable> {
        public void configure(JobConf job){
            this.blockSize = Integer.parseInt(job.get("block_size"));
            this.isFirstIteration = (Integer.parseInt(job.get("stage")) == 0);
            this.Agregate = 0.0;
            nbModel = 0;
            Somm_Residual_ALL = new Hashtable<String,Double>(); //pour garder trace de la selection et faire correspondre dans H


            Y_Prob_ALL = new Hashtable<String,DenseVector>();
            H_ALL = new Hashtable<String,DenseVector>();
            this.nbTest = Integer.parseInt(job.get("nbTest"));
            this.nbBagIter = Integer.parseInt(job.get("Bag_Iter"));

            Somm_Y_ALL = new DenseVector(nbTest);
            H_ALL = new Hashtable<String, DenseVector>();
            this.TestPath = job.get("test_path");

        }


        public void reduce(Text key, Iterator<TypedBytesWritable> values,
                           OutputCollector<Text,TypedBytesWritable> output,
                           Reporter reporter)
                throws IOException {
            if (this.output == null) {
                this.output = output;
            }

            is_reduce = true;
            numReduce = 2;

            while (values.hasNext()) {
                //output.collect(key,values.next());
                if (key.toString().startsWith("H")) //Récuperer les modèles
                {
                    double[] H_Model = decodeTypedBytesArray(values.next());
                    nbColumns = H_Model.length;
                    H_ALL.put(key.toString(),new DenseVector(H_Model));

                }
                else
                {
                    output.collect(key,values.next());
                }


            }
        }
    }

}







