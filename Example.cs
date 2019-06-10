using System.IO;

using ML.Data;
using ML.LinearRegression;

namespace Example
{
    class Program
    {
        static void Main()
        {
            Data data1 = new Data();
            data1.SaveToFile("sample1.txt");

            Data data2 = new Data("sample1.txt");

            Model model1 = new Model(data2);
            model1.SaveToFile("model1.txt");

            Model model2 = new Model("model1.txt");

            data2.Generate();
            data2.SaveToFile("sample2.txt");

            model2.Training(data2);
            model2.SaveToFile("model2.txt");

            StreamWriter file = new StreamWriter("result1.txt");

            file.WriteLine(model1.QualityFunc(data1));
            file.WriteLine(model1.CrossValid(data1));

            file.Close();

            file = new StreamWriter("result2.txt");

            file.WriteLine(model2.QualityFunc(data1));
            file.WriteLine(model2.CrossValid(data2));

            file.Close();
        }
    }
}

/*
 * Example:
 * - подключение Data и LinearRegression
 * - генерация выборки №1 по-умолчанию с сохранением в файл
 * - создание выборки №2 из сохраненной выборки №1
 * - создание и обучение модели №1 по выборке 2 с сохранением в файл
 * - создание модели №2 из файла модели №1
 * - повторная генерирация по-умолчанию выборки №2 с сохранением в файл
 * - обучение модели №2 по выборке №2 с сохранением в файл
 * - вычисление функционала качества и скользящего контроля для модели №1 по выборке №1 с сохранением в файл
 * - вычисление функционала потерь модели №2 по выборке №1 и скользящего контроля модели №2 по выборке №2 с сохранением в файл
*/

