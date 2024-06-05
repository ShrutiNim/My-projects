// Customised Database Management System

class Program418
{
    public static void main(String Arg[])
    {
        DBMS obj = new DBMS();

        obj.InsertIntoTable("Amit","Pune",89);
        obj.InsertIntoTable("Pooja","Mumbai",95);
        obj.InsertIntoTable("Gauri","Pune",90);
        obj.InsertIntoTable("Rahul","Satara",80);
        obj.InsertIntoTable("Neha","Pune",78);

        obj.SelectStarFrom();

        obj.SelectStarFromWhereCity("Pune");

        obj.SelectCount();

        obj.SelectSum();

        obj.SelectAvg();

        obj.SelectMax();

        obj.SelectMin();

        obj.SelectStarFromWhereName("Amit");

        obj.UpdateCity(3,"Nashik");

        obj.SelectStarFrom();

        obj.DeleteFrom(5);

        obj.SelectStarFrom();

        obj.InsertIntoTable("Rukmini","Kolhapur",77);

        obj.SelectStarFrom();
    }
}

class node
{
    private static int Counter = 1;

    public int Rno;
    public String Name;
    public String City;
    public int Marks;

    public node next;

    public node(String B, String C, int D)
    {
        Rno = Counter;
        Counter++;

        Name = B;
        City = C;
        Marks = D;
        next = null;
    }
}

class DBMS
{
    private node first;

    public DBMS()
    {
        first = null;
        System.out.println("DBMS initialised successfully");
        System.out.println("Student table gets created successfully...");
    }

    public void InsertIntoTable(String Name, String City, int Marks)
    {
        node newn = new node(Name, City, Marks);
        
        if(first == null)
        {
            first = newn;
        }
        else
        {
            node temp = first;
            while(temp.next != null)
            {
               temp = temp.next;
            }

            temp.next = newn;

            System.out.println("One record gets inserted successfully");
        }
    }

    public void SelectStarFrom()
    {
        System.out.println("Data from the student table is : ");

        node temp = first;

        System.out.println("---------------------------------------------------------------------");

        while(temp != null)
        {
            System.out.println(temp.Rno+"\t"+temp.Name+"\t"+temp.City+"\t"+temp.Marks);
            temp = temp.next;
        }
        System.out.println("---------------------------------------------------------------------");
    }

    public void SelectStarFromWhereCity(String str)
    {
        System.out.println("Data from the student table where city is : " + str);

        node temp = first;

        System.out.println("---------------------------------------------------------------------");

        while(temp != null)
        {
            if(str.equals(temp.City))
            {
                System.out.println(temp.Rno+"\t"+temp.Name+"\t"+temp.City+"\t"+temp.Marks);
            }
            temp = temp.next;
        }
        System.out.println("---------------------------------------------------------------------");
    }

    public void SelectStarFromWhereName(String str)
    {
        System.out.println("Data from the student table where name is : " + str);

        node temp = first;

        System.out.println("---------------------------------------------------------------------");

        while(temp != null)
        {
            if(str.equals(temp.Name))
            {
                System.out.println(temp.Rno+"\t"+temp.Name+"\t"+temp.City+"\t"+temp.Marks);
            }
            temp = temp.next;
        }
        System.out.println("---------------------------------------------------------------------");
    }

    public void SelectCount()
    {
        node temp = first;
        int iCnt = 0;

        while(temp != null)
        {
            iCnt++;
            temp = temp.next;
        }

        System.out.println("Number of records in the table : " + iCnt);
    }

    public void SelectSum()
    {
        node temp = first;
        int iSum = 0;

        while(temp != null)
        {
            iSum = iSum + temp.Marks;
            temp = temp.next;
        }

        System.out.println("Summation of Marks column is : " + iSum);
    }

    public void SelectAvg()
    {
        node temp = first;
        int iAvg = 0, iSum = 0, iCnt = 0;

        while(temp != null)
        {
            iSum = iSum + temp.Marks;
            temp = temp.next;
            iCnt++;
        }

        System.out.println("Average of Marks column is : " + (float)(float)iSum/(float)iCnt);
    }

    public void SelectMax()
    {
        node temp = first;
        int iMax = 0;

        if(temp != null)
        {
            iMax = temp.Marks;
        }

        while(temp != null)
        {
            if(iMax < temp.Marks)
            {
                iMax = temp.Marks;
            }
            temp = temp.next;
        }

        System.out.println("Maximum from Marks column is : " + iMax);
    }

    public void SelectMin()
    {
        node temp = first;
        int iMin = 0;

        if(temp != null)
        {
            iMin = temp.Marks;
        }

        while(temp != null)
        {
            if(iMin > temp.Marks)
            {
                iMin = temp.Marks;
            }
            temp = temp.next;
        }

        System.out.println("Minimum from Marks column is : " + iMin);
    }

    public void UpdateCity(int no, String str)
    {
        node temp = first;

        while(temp != null)
        {
            if(temp.Rno == no)
            {
                temp.City = str;
                break;
            }
            temp = temp.next;
        }

        System.out.println("Record gets updated...");
    }

    public void DeleteFrom(int no)
    {

        node temp = first;

        if(temp == null)
        {
            return;
        }

        if(temp.Rno == no)
        {
            first = first.next;
        }

        while(temp.next != null)
        {
            if(temp.next.Rno == no)
            {
                temp.next = temp.next.next;
                break;
            }
            temp = temp.next;
        }
    }
}


/*
 * Supperted Queries
 *  
 * 1 : insert into student values('Amit','Pune',78);
 * 2 : select * from student;
 * 3 : select * from student where City = 'Pune';
 * 4 : select Count(Marks) from student;
 * 5 : selectcount
 * 6 : selectsum
 * 7 : selectavg
 * 
 */