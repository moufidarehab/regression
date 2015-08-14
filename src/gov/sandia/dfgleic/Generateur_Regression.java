package gov.sandia.dfgleic;

//import Jama.Matr
import java.net.UnknownHostException;
import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: hubiquitus
 * Date: 10/1/13
 * Time: 2:41 PM
 * To change this template use File | Settings | File Templates.
 */
public  class Generateur_Regression {

    public static void main(String[] args) throws UnknownHostException
    {

        long startTime = System.currentTimeMillis();


        int col=100; int lign=500000;

        double[][] regression=new double [lign][col+1];
       double [] y=new double [lign];
        Random randomGenerator = new Random();
        double randomval;

        for( int i=0;i<regression.length ;i++)
        {

            for( int j=0; j<col+1;j++)
            {
                if (j==0)
                {
                    regression[i][0]=1;
                }
                else
                {
                    randomval = randomGenerator.nextInt(900);
                    regression[i][j]=randomval;
                }
            }
        }
 /*           System.out.println("**********la matrice des X************************");
            for( int l=0;l<regression.length ;l++)
            {
                for( int j=0; j<col;j++)
                {
                    System.out.println(regression[l][j])  ;
                }
                System.out.println("\n");
            }
   */
        for( int i=0;i<y.length ;i++)
        {
            randomval = randomGenerator.nextInt(900);
            y[i]=randomval;
        }

       // regression_class regr = new regression_class (regression, y);
        //for (int j=0;j<regr.beta.getRowDimension();j++)
       //     System.out.println("B"+j+"="+ regr.beta(j));

        long endTime = System.currentTimeMillis();
        System.out.println("temps execution= "+ (endTime-startTime));

    }
}