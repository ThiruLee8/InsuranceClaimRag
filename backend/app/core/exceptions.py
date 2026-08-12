from fastapi import HTTPException


class AppError(Exception):
    def __init__(self, message: str, error_code: str, status_code: int = 400):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(message)


def raise_http(message: str, error_code: str, status_code: int = 400) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={
            "success": False,
            "message": message,
            "errorCode": error_code,
        },
    )
