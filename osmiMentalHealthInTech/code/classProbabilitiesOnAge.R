#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

#imports

library(shiny)
library(nnet)

#constants
lineWidth = 3

#load in model
#and load in dataset
givenMod.mc = readRDS("../models/finalClusterAssignmentClassifier.rds")
refDataFrame = read.csv("../data/processed/clusterData_inference.csv")
genderDataFrame = read.csv("../data/preprocessed/genderCountFrame.csv")

# Define UI for application that draws a histogram
ui <- fluidPage(
   # Application title
   titlePanel("Predicted Class Probabilities On Age"),
   
   # Sidebar with a slider input for number of bins 
   sidebarLayout(
      sidebarPanel(
         selectInput("diagnosedWithMHD",
                     "Diagnosed with Mental Health Condition:",
                     choices = unique(refDataFrame$diagnosedWithMHD)),
         selectInput("gender",
                     "Gender of the Respondent:",
                     choices = unique(genderDataFrame$genderMap)),
         selectInput("companySize",
                     "Size of Respondent's Employer:",
                     choices = unique(refDataFrame$companySize))
      ),
      
      # Show a plot of the generated distribution
      mainPanel(
         plotOutput("classPredictedProbPlot",height = "600px")
         
      )
   )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
   
   output$classPredictedProbPlot <- renderPlot({
      #get prediction frame
      ageRange = 21:40
      predictFrame = data.frame(diagnosedWithMHD = input$diagnosedWithMHD,
                                gender = input$gender,
                                companySize = input$companySize,
                                isUSA = 1,age = ageRange)
      #then make predictions
      predictions = predict(givenMod.mc,newdata = predictFrame,type = "probs")
      #then export that to a reasonable-looking data frame
      predictFrame$classOneProb = predictions[,1]
      predictFrame$classTwoProb = predictions[,2]
      predictFrame$classThreeProb = predictions[,3]
      #then plot partial effects
      #then plot these partial effects plots
      plot(x = NULL,y = NULL,xlim = c(21,40),ylim = c(0,1),
           xlab = "Age",ylab = "Predicted Probability",
           main = "Predicted Class Assignment\nOn Age")
      lines(predictFrame$age,predictFrame$classOneProb,col = "Blue",
            lwd = lineWidth)
      lines(predictFrame$age,predictFrame$classTwoProb,col = "Red",
            lwd = lineWidth)
      lines(predictFrame$age,predictFrame$classThreeProb,col = "Green",
            lwd = lineWidth)
      legend(21,1,legend = c("Class 1","Class 2","Class 3"),
             col = c("Blue","Red","Green"),lty = 1,lwd = lineWidth)
   })
}

# Run the application 
shinyApp(ui = ui, server = server)

