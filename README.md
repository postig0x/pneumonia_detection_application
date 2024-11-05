# Pneumonia Detection Application

## Purpose

 The purpose of this deployment is to develop a robust and secure machine learning application that assists healthcare professionals in diagnosing pneumonia from X-ray images. The Flask application leverages a fine-tuned version of the pre-trained ResNet50 Convolutional Neural Network to analyze X-ray images and provide predictions on the presence of pneumonia. This automation aims to enhance the diagnostic process, allowing doctors to make informed decisions more quickly.

### Security

The web, application, and model training servers reside in a private subnet, using Nginx in the public subnet as a reverse proxy to the web server. This decision highlights the importance of security in cloud-based applications, ensuring that sensitive medical data and the application are not directly exposed to the internet.

A monitoring server with Prometheus and Grafana is provisioned in a public subnet for monitoring the model training server's performance.

_System diagram is shown below._

## Steps

The application is designed to allow x-ray images to be uploaded via the frontend, processed by a neural network model in the backend, and the results stored in Redis to be displayed on the UI.

In order to provision the infrastructure discussed and build a working application, the following steps were taken:

1. Create `t3.medium` instance for terraform
    1. [Install](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) terraform
2. Build out the infrastructure using `terraform/main.tf` with the `terraform` `init`, `validate`, `plan`, and `apply` commands.
3. Configure the connections in the backend and frontend:
    1. **Monitoring Server**: Add a `node_exporter` job to the monitoring server's `/opt/prometheus/prometheus.yml` file. Restart the prometheus daemon.
3. **Nginx Server**: Configure Nginx to route to the web server by modifying `/etc/nginx/sites-enabled/default` on the Nginx server. Restart Nginx daemon
4. **App Server**: Configure redis connection by modifying `/etc/redis/redis.conf`
5. **ML Training Server**: Modify the project's `/model/inference.py` file to connect to the application server's IP. This allows the model training server to connect to the database on the application server and post the results of the inferences on the test images to the database. Doing this allows the frontend to access that data and display the results.
    1. After modifying that file, the model can be compiled and trained using `model/cnn.py`. The output is a `.keras` file that can be passed back to the application server, which can run inference using that file on the CPU.
    2. Run `model/inference.py` _AFTER_ the step above.
6. Set up python environment and install dependencies on the application server's project folder (`pneumonia_api/`) and run the server with `nohup gunicorn --bind 0.0.0.0:5000 app:app &` so that it runs in the background.
7. Connect to the UI (web) server and set `API_URL` to the private IP of the application server in `pneumonia_web/app.py`. Set up python environment and install dependencies inside `pneumonia_web/` and run the server in the background: `nohup gunicord --config gunicorn_config.py app:app &`.

### Improving Model Performance

#### Fine-tuned Model

In order to fine-tune the model to improve it's previous performance (all images were detected as positive for pneumonia), it's imperative to reduce overfitting. A few methods were implemented to achieve this, including adding L2 and Dropout regularization on each Dense sequential layer added to the base Keras ResNet50 model. Batch normalization is introduced to each Dense layer as well, so that the outputs of each layer do not shift the input distribution for the next layers. The last 15 layers of the ResNet50 model are also unfrozen to focus the model on binary classification as opposed to ResNet's default multi-class classification.

#### Loss Function

We want to select the appropriate loss function for this application (binary image classification). Potential options include **Binary Cross-Entropy Loss** and **Focal Loss**, but the former was chosen because it is primarily focused on binary classification. Focal loss is useful for imbalanced datasets, which did not seem to be an issue for this workload.

#### Optimizers

The common `adam` (adaptive moment estimation) optimizer was chosen, but notable options include:

- `sgd` - stochastic gradient descent
- `nadam` - combinatio of `adam` and `nesterov` - may provide better convergence
- `adadelta` - reduces aggressive, monotonically decreasing learning rate of `adagrad` optimizer

#### Metrics

The following metrics were considered:

- `accuracy` - proportion of correct predictions (not for imbalanced data)
- `precision` - ratio of true positive predictions to total predicted positives. useful when the  cost of false positives is high
- `recall` (sensitivity) - ratio of true positive predictions to total actual positives. useful when cost of false negatives is high
- `F1 Score` - balance of accuracy and precision

 ```python
 from keras.metrics import Precision
 metrics=[Precision(), Recall()]
 ```

- `AUC-ROC` - area under the receiver operating characteristic curve. evaluates model's ability to distinguish between classes across different thresholds. useful for binary classification problems

 ```python
 from keras.metrics import AUC
 metrics=[AUC()]
 ```

For pneumonia predictions, the cost of **false negatives** is typically considered much higher than false positives, making `recall` the better metric option.

## Screenshots

![frontend](/screenshots/frontend.png)

![monitoring](/screenshots/monitoring.png)

## System Design Diagram

![ai_wl_diagram](/screenshots/ai_wl_diagram.png)

## Optimization

The model can be further optimized by utilizing a DAG (directed acyclic graph) based model, with layers such as `Conv2D`, instead of utilizing the Sequential Dense layers already provided to us as a starting point. The reason for this is that a DAG model may perform better due to it's efficiency and ability to capture both sequential and non-sequential relationships, which may prove to be beneficial when classifying images.

There can be further optimization in automation by adding commands to the `user_data` scripts (`terraform/*.sh`) that automatically replace the necessary IP addresses in configuration files, and creating virtual environments and launching the servers. This also applies to automating model training and running inference on images to update the application data base (`model/{cnn,inference}.py`). It would be important to have a shutdown command on the model server's `p3.2xlarge` instance so as not to accrue unneccessary costs.

## Conclusion

In conclusion, the deployment of the machine learning application for pneumonia diagnosis represents a significant advancement in healthcare technology, leveraging a fine-tuned ResNet50 model to enhance diagnostic accuracy and speed. The careful consideration of security measures, including the use of private subnets and monitoring tools, ensures the protection of sensitive medical data. Additionally, the implementation of various optimization techniques, such as regularization and appropriate loss functions, has improved the model's performance, making it a reliable tool for healthcare professionals. Future enhancements could focus on further automation and optimization of the infrastructure, ultimately leading to a more efficient and effective diagnostic process. This project not only demonstrates the potential of AI in medicine but also sets a foundation for future innovations in healthcare applications.
