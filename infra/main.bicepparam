using './main.bicep'

// Development environment parameters
param environment = 'dev'
param location = 'westus2'
param baseName = 'langgraph-agent'
param modelDeploymentName = 'gpt-4o'
param containerImage = ''
