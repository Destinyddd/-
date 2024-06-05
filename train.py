from model import *

def optimize_model_parameters(network, training_data, training_labels, validation_data, validation_labels, lr_initial, lr_decay, regularisation_strength, total_epochs, samples_per_batch, activation_fn='relu'):
    train_size = training_data.shape[0]
    number_of_batches = train_size // samples_per_batch

    history_loss_train = []
    history_accuracy_train = []
    history_loss_val = []
    history_accuracy_val = []
    highest_val_accuracy = 0.0
    optimal_model = network

    for epoch in range(total_epochs):
        shuffled_indices = np.random.permutation(train_size)
        training_data_shuffled = training_data[shuffled_indices]
        training_labels_shuffled = training_labels[shuffled_indices]

        for batch_index in range(number_of_batches):
            start_index = batch_index * samples_per_batch
            end_index = start_index + samples_per_batch
            batch_data = training_data_shuffled[start_index:end_index]
            batch_labels = training_labels_shuffled[start_index:end_index]
            predictions, fwd_cache = model_forward(network, batch_data, activation_fn)
            gradients = model_backward(network, fwd_cache, batch_data, batch_labels, predictions, regularisation_strength, activation_fn)
            network = update_parameters(network, gradients, lr_initial)

        # Validate and Evaluate Training
        predictions_train, _ = model_forward(network, training_data, activation_fn)
        loss_train = calculate_loss(training_labels, predictions_train, network, regularisation_strength)
        accuracy_train = calculate_accuracy(predictions_train, training_labels)
        
        # Validate and Evaluate Validation Set
        predictions_val, _ = model_forward(network, validation_data, activation_fn)
        loss_val = calculate_loss(validation_labels, predictions_val, network, regularisation_strength)
        accuracy_val = calculate_accuracy(predictions_val, validation_labels)
        
        if highest_val_accuracy < accuracy_val:
            highest_val_accuracy = accuracy_val
            optimal_model = network

        history_loss_train.append(loss_train)
        history_accuracy_train.append(accuracy_train)
        history_loss_val.append(loss_val)
        history_accuracy_val.append(accuracy_val)
        lr_initial *= lr_decay

        print(f'Epoch {epoch+1} of {total_epochs}: Train Loss = {loss_train:.4f}, Train Accuracy = {accuracy_train:.4%}; Val Loss = {loss_val:.4f}, Val Accuracy = {accuracy_val:.4%}')

    return network, optimal_model, history_loss_train, history_accuracy_train, history_loss_val, history_accuracy_val