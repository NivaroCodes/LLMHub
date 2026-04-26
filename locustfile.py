from locust import HttpUser, task, between

class LLMHubUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def chat(self):
        self.client.post(
            "/chat",
            json={"message": "Расскажи про Сан-Франциско"}
        )