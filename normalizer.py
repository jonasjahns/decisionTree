# This function is used to create a test sample with all the denied transactions
# Since we have more than 200 approved transactions to each denied transaction
# We had to made a train base with enough denied transactions for the algorithm to work


def df_sample(df):
    df_denied = df.query('ID_STATUS_MENSAGEM == 3')
    df_approved = df.query('ID_STATUS_MENSAGEM == 2').sample((len(df_denied) * 3))
    return df_denied.append(df_approved)
