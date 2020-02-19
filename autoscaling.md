# Lab: Autoscaling Amazon SageMaker endpoints

In this lab, we'll use AWS Console to setup Amazon Sagemaker endpoint and then
configure its autoscaling

## Select a trained model

![](./images/autoscaling/select-model.png)

## Create an endpoint configuration

![](./images/autoscaling/create-endpoint-conf.png)
![](./images/autoscaling/create-endpoint-2.png)

## Create an Endpoint

Click on "Create endpoint" ![](./images/autoscaling/endpoint-conf.png) Then,
create and configure endpoint ![](./images/autoscaling/create-endpoint.png)

## Configure autoscaling

Click on the endpoint to open it's settings page
![](./images/autoscaling/click-endpoint.png) then, scroll down to "Endpoint
runtime settings" to select the variant and then click on "Configure
autoscaling" ![](./images/autoscaling/select-runtime-settings.png) Now,
configure autoscaling properties - for this example I have set "Maximum instance
count" to 2 and "Target value" to 100 for the target metric
"SageMakerVariantInvocationPerInstance" (the average number of times per minute
that each instance for a variant is invoked), then clicked on "Save"
![](./images/autoscaling/configure-asg.png)

[< Home](./readme.md)
