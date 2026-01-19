// Simplified Azure Bicep template for AI Agent Infrastructure
// Uses existing Azure OpenAI resource

targetScope = 'resourceGroup'

// ============================================
// Parameters
// ============================================

@description('Environment name (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

@description('Azure region for resources')
param location string = resourceGroup().location

@description('Base name for resources')
param baseName string = 'langgraph-agent'

@description('Container image to deploy')
param containerImage string = ''

@description('Existing Azure OpenAI endpoint')
param existingAoaiEndpoint string

@description('Existing Azure OpenAI API key')
@secure()
param existingAoaiKey string

@description('Azure OpenAI model deployment name')
param modelDeploymentName string = 'gpt-4o'

// ============================================
// Variables
// ============================================

var resourceSuffix = '${baseName}-${environment}-${uniqueString(resourceGroup().id)}'
var shortSuffix = '${baseName}-${environment}'
var tags = {
  Environment: environment
  Application: 'LangGraph-Agent'
  ManagedBy: 'Bicep'
}

// ============================================
// Container Apps Environment
// ============================================

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'logs-${shortSuffix}'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appins-${shortSuffix}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${shortSuffix}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    daprAIInstrumentationKey: appInsights.properties.InstrumentationKey
  }
}

// ============================================
// Container App - AI Agent
// ============================================

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-${shortSuffix}'
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8080
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
      }
      secrets: [
        {
          name: 'aoai-key'
          value: existingAoaiKey
        }
        {
          name: 'appinsights-connectionstring'
          value: appInsights.properties.ConnectionString
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'langgraph-agent'
          image: !empty(containerImage) ? containerImage : 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: existingAoaiEndpoint
            }
            {
              name: 'AZURE_OPENAI_API_KEY'
              secretRef: 'aoai-key'
            }
            {
              name: 'AZURE_OPENAI_DEPLOYMENT_NAME'
              value: modelDeploymentName
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              secretRef: 'appinsights-connectionstring'
            }
            {
              name: 'MAX_REFLECTION_ITERATIONS'
              value: '3'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
}

// ============================================
// Outputs
// ============================================

output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output containerAppName string = containerApp.name
output appInsightsName string = appInsights.name
output logAnalyticsName string = logAnalytics.name
