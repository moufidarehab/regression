package gov.sandia.dfgleic;
import java.util.*;

public class RandomIndexOfMax {

    public RandomIndexOfMax() {
    }

    public int getIndex(double Array[]) {
        double Max = Array[0];

        for (int i=1; i<Array.length; i++)
            if (Array[i] > Max)
                Max = Array[i];

        int Count = 0;
        for (int i=0; i<Array.length; i++)
            if (Array[i] == Max)
                Count++;

        Random Rand = new Random();
        int Choose = Rand.nextInt(Count)+1;

        Count = 0;
        for (int i=0; i<Array.length; i++)
        {
            if (Array[i] == Max)
                Count++;
            if (Count == Choose)
                return i;
        }

        return -1;
    }
}
