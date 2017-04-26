import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

fun main(args: Array<String>) {

    //number of rows and columns in the input pictures
    val numRows = 28
    val numColumns = 28
    val outputNum = 10 // number of output classes
    val batchSize = 128 // batch size for each epoch
    val rngSeed = 123 // random number seed for reproducibility
    val numEpochs = 15 // number of epochs to perform

    val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)

    val conf = NeuralNetConfiguration.Builder()
            .seed(rngSeed) //include a random seed for reproducibility
            // use stochastic gradient descent as an optimization algorithm
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.006) //specify the learning rate
            .updater(Updater.NESTEROVS).momentum(0.9) //specify the rate of change of the learning rate.
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, DenseLayer.Builder() //create the first, input layer with xavier initialization
                    .nIn(numRows * numColumns)
                    .nOut(1000)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .layer(1, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                    .nIn(1000)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .pretrain(false).backprop(true) //use backpropagation to adjust weights
            .build()

    val model = MultiLayerNetwork(conf)
    model.init()

    //print the score with every 100 iterations
    model.setListeners(ScoreIterationListener(100))

    println("Train model....")
    for (i in 0..numEpochs - 1) {
        model.fit(mnistTrain)
    }

    println("Evaluate model....")
    val eval = Evaluation(outputNum) //create an evaluation object with 10 possible classes
    while (mnistTest.hasNext()) {
        val next = mnistTest.next()
        val output = model.output(next.featureMatrix) //get the networks prediction
        eval.eval(next.labels, output) //check the prediction against the true class
    }

    println(eval.stats())

}