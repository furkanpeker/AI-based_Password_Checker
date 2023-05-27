import paramiko

def fetchCredentials():
    # copy the file which has the users' identifies from the RADIUS database:
    # Getting the identifies
    ssh_session = paramiko.SSHClient()
    hostname = input('Hostname: ')
    username = input('Username: ')
    password = input('password: ')
    port = int(input('Port: '))
    # Establishing the connection
    ssh_session.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_session.connect(hostname = hostname, username = username, password = password, port = port)
    # Implementation phase
    sftp_client = ssh_session.open_sftp()
    # fetching the file from the server:
    sftp_client.get('/usr/local/etc/raddb/users', 'users.txt')
    ssh_session.close()

def getPasswords():
    # open the users.txt and converting its content into a list " "
    f = open('users.txt', 'r')
    data = f.read()
    # replacing end of line('/n') with ' ' and
    # splitting the text it further when '.' is seen.
    data_into_list = data.replace('\n', ' ').split(" ")
    print(data_into_list)
    f.close()

    passwords = []

    passwords.append(data_into_list[4])
    i=12
    while (i < len(data_into_list)):
        passwords.append(data_into_list[i])
        i+=8
    print(passwords)

fetchCredentials()
getPasswords()
