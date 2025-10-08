pipeline{
    agent any 

    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning GitHub repo to Jenkins............'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/rishabh16-2005/Reservation-Prediction-.git']])
                }
            }
        }
    }
}