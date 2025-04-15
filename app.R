library(shiny)
library(ggplot2)
library(dplyr)
library(stringr)
library(tidyr)

# 📦 Chargement des données exportées depuis Python
data <- read.csv("dataset_for_shiny.csv", stringsAsFactors = FALSE)

# 📤 Explosion des événements par texte
data_events <- data %>%
  mutate(text_id = row_number()) %>%
  separate_rows(events, sep = ",\\s*") %>%
  filter(events != "")

# 🖼️ UI
ui <- fluidPage(
  titlePanel("📊 Analyse des événements dans les textes"),
  
  sidebarLayout(
    sidebarPanel(
      checkboxGroupInput("event_filter", "Filtrer les événements :", 
                         choices = unique(data_events$events),
                         selected = unique(data_events$events))
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Répartition des événements",
                 plotOutput("eventPlot")),
        
        tabPanel("Longueur des textes",
                 plotOutput("lengthPlot")),
        
        tabPanel("Événements par texte",
                 plotOutput("eventCountPlot")),
        
        tabPanel("Comparaison des groupes",
                 selectInput("metric", "Choisir la métrique à comparer :", 
                             choices = list(
                               "Longueur des textes" = "n_tokens",
                               "Nombre d'événements" = "n_events",
                               "Diversité des événements" = "diversity")),
                 plotOutput("groupComparisonPlot"),
                 verbatimTextOutput("statTestResult"))
      )
    )
  )
)

# 🧠 SERVER
server <- function(input, output) {
  
  # 🔍 Données filtrées par événements
  filtered_data <- reactive({
    data_events %>% filter(events %in% input$event_filter)
  })
  
  # 📊 Répartition des événements
  output$eventPlot <- renderPlot({
    req(nrow(filtered_data()) > 0)
    
    filtered_data() %>%
      count(events, sort = TRUE) %>%
      ggplot(aes(x = reorder(events, -n), y = n)) +
      geom_bar(stat = "identity", fill = "skyblue") +
      theme_minimal() +
      labs(title = "📊 Répartition des types d’événements", x = "Événement", y = "Occurrences") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  # ✂️ Histogramme de longueur de texte
  output$lengthPlot <- renderPlot({
    req(nrow(data) > 0)
    
    ggplot(data, aes(x = n_tokens)) +
      geom_histogram(fill = "lightgreen", bins = 30) +
      theme_minimal() +
      labs(title = "✂️ Distribution de la longueur des textes", x = "Nombre de tokens", y = "Fréquence")
  })
  
  # 🧩 Histogramme du nombre d’événements par texte
  output$eventCountPlot <- renderPlot({
    req(nrow(data_events) > 0)
    
    data_events %>%
      group_by(text_id) %>%
      summarise(event_count = n()) %>%
      ggplot(aes(x = event_count)) +
      geom_histogram(fill = "coral", bins = 20) +
      theme_minimal() +
      labs(title = "🧩 Nombre d’événements par texte", x = "Nb d'événements", y = "Nb de textes")
  })
  
  # 📐 Données pour comparaison de groupes
  grouped_data <- reactive({
    dominant_event_df <- data_events %>%
      count(text_id, events) %>%
      group_by(text_id) %>%
      slice_max(n, n = 1, with_ties = FALSE) %>%
      ungroup()
    
    event_counts <- data_events %>%
      group_by(text_id) %>%
      summarise(n_events = n(),
                diversity = n_distinct(events))
    
    data %>%
      mutate(text_id = row_number()) %>%
      left_join(event_counts, by = "text_id") %>%
      left_join(dominant_event_df, by = "text_id") %>%
      filter(!is.na(events))
  })
  
  # 📊 Boxplot comparatif
  output$groupComparisonPlot <- renderPlot({
    req(input$metric)
    df <- grouped_data()
    req(nrow(df) > 0, input$metric %in% names(df))
    
    ggplot(df, aes(x = events, y = .data[[input$metric]])) +
      geom_boxplot(fill = "skyblue") +
      theme_minimal() +
      labs(title = paste("Comparaison par événement dominant :", input$metric),
           x = "Événement dominant", y = input$metric) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  # 🧪 Test statistique
  output$statTestResult <- renderPrint({
    req(input$metric)
    df <- grouped_data()
    req(nrow(df) > 0, input$metric %in% names(df))
    
    kruskal.test(as.formula(paste(input$metric, "~ events")), data = df)
  })
}

# 🚀 Lancement de l'app
shinyApp(ui = ui, server = server)
