class ImageEncoder:
  def __init__(self, choice):
    # TODO: Define feature_extract

    self.input = CROPPED_WIDTH
    # Detect if we have a GPU available
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.trained = False
    self.dropout = nn.Dropout(p=0.5)
    self.ensemble = None
    if choice == "GoogLeNet":
      # GoogLeNet
      self.model_name = "GoogLeNet_Inception_v3"
      model_ft = models.inception_v3(pretrained=True)

      # Check whether we need this: Handle the auxilary net
      num_ftrs = model_ft.AuxLogits.fc.in_features
      model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

      # Handle the primary net
      num_ftrs = model_ft.fc.in_features
      model_ft.fc = nn.Linear(num_ftrs,num_classes)
      
      self.transform = transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
      model_ft = model_ft.to(device)
      self.model = model_ft

    elif choice == "AlexNet":
      self.model_name = "AlexNet"
      model_ft = models.alexnet(pretrained=True)
      self.model = model_ft.to(self.device)

    #TODO
    elif choice == "VGG-19":
      self.model_name = "VGG-19"
      model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
      self.model = model_ft.to(self.device)

  def generate_ensembling_models(self, num_models):
    # https://pytorch.org/functorch/0.1.0/notebooks/ensembling.html
    models = [self.model for _ in range(num_models)]
    fmodel, params, buffers = combine_state_for_ensemble(models)
    self.ensemble = [fmodel, params, buffers]
  
  def train_and_val_model(self,num_epochs, trainloader, validloader, lr=0.001, momentum=0.9, loss="CE"):
    """https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html"""
    save_file_name = f"{self.model_name}_{num_epochs}_{lr}_{momentum}"
    early_stopping = 5
    # use SGD with learning rate = 0.001 and momentum = 0.9
    self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
    if loss=="CE":
      criterion = nn.CrossEntropyLoss()
    elif loss=="L1":
      criterion = nn.L1Loss()
    else:
      criterion = nn.MSELoss()
    valid_loss_min = np.Inf
    epochs_no_improve, valid_best_acc = 0, 0
    outputs_losses = []
    losses_colums = ["training_loss","validation_loss"]
    outputs_acc = []
    acc_columns = ["training_acc","validation_acc"]
    for epoch in tqdm(range(num_epochs)):
      training_loss, validation_loss, testing_loss = 0, 0, 0
      training_acc, validation_acc, testing_acc = 0, 0, 0
      for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        self.optimizer.zero_grad()
        output = self.model(images)
        loss = criterion(output, labels)
        loss.backward()
        self.optimizer.step()
        training_loss += loss.item() * images.size(0)
        # Get training accuracy
        _, pred = torch.max(output, dim=1)
        correct_tensor = pred.eq(labels.data.view_as(pred))
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
        training_acc += accuracy.item() * images.size(0)
      else:
        with torch.no_grad():
          self.model.eval()
          print("model start validating")
          for images, labels in validloader:
            images = images.view(images.shape[0], -1)
            output = self.model(images)
            loss = criterion(output, labels)
            validation_loss += loss.item() * images.size(0)
            # Get Validation accuracy
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(
                correct_tensor.type(torch.FloatTensor))
            validation_acc += accuracy.item() * images.size(0)
       # Calculate average losses
        train_loss = training_loss / len(trainloader.dataset)
        valid_loss = validation_loss / len(validloader.dataset)
        # Calculate average accuracy
        train_acc = training_acc / len(trainloader.dataset)
        valid_acc = validation_acc / len(validloader.dataset)
        outputs_losses.append([
            train_loss, valid_loss
        ])
        outputs_acc.append([
            train_acc, valid_acc
        ])
        # Save the model if validation loss decreases
        if valid_loss < valid_loss_min:
            # Save model
            torch.save(self.model.state_dict(), save_file_name)
            # Track improvement
            epochs_no_improve = 0
            valid_loss_min = valid_loss

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping:
                print("Early stopping the model.")
                self.model.load_state_dict(torch.load(save_file_name))
                self.model.optimizer = self.optimizer
                outputs_losses = pd.DataFrame(
                    outputs_losses,
                    columns=losses_colums
                )
                outputs_acc = pd.DataFrame(
                    outputs_acc,
                    columns=acc_columns
                )
                # self.model = model
                self.trained = True
                return self.model, outputs_losses, outputs_acc
    outputs_losses = pd.DataFrame(
        outputs_losses,
        columns=losses_colums
    )
    outputs_acc = pd.DataFrame(
        outputs_acc,
        columns=acc_columns
    )
    # self.model = model
    self.trained = True
    return self.model, outputs_losses, outputs_acc
  
  def forward(self, images, num_models, dropout=False, ensemble=False):
    # think about the ensemble and dropout a bit more
    if ensemble and self.ensemble == None:
      self.generate_ensembling_models(num_models)
    if dropout and ensemble:
      minibatches = images[:num_models]
      model_outputs = self.dropout(vmap(self.ensemble[0])(self.ensemble[1], self.ensemble[2], minibatches))
    elif dropout:
      model_outputs = self.dropout(self.model(images))
    elif ensemble:
      minibatches = images[:num_models]
      model_outputs = vmap(self.ensemble[0])(self.ensemble[1], self.ensemble[2], minibatches)
    else:
      model_outputs = self.model(images)
    # might not want embeddings?
    #embeddings = self.embed(model_outputs)
    return model_outputs

  def encode_image(self, image_path, normalize=False):
    """
    have to make this so that this can also process a lot of images at the same time
    """
    img = Image.open(image_path)
    img_processed = self.transform(img).unsqueeze(0)
    if torch.cuda.is_available():
      img_processed = img_processed.to('cuda')
      self.model.to('cuda')

    if not self.trained:
      print("Model has not been trained")
      return -1
    with torch.no_grad():
      output = self.model(img_processed)[0]
    if normalize:
      # add normalization code here
      output = torch.nn.functional.softmax(output, dim=0)
    return output