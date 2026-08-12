import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { ChatService } from './chat.service';
import { environment } from '../../../environments/environment';

describe('ChatService', () => {
  let service: ChatService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ChatService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('posts a chat question', () => {
    service.ask('What caused the loss?', 'conv-1').subscribe((res) => {
      expect(res.answer).toContain('pipe');
      expect(res.sources.length).toBe(1);
    });

    const req = http.expectOne(`${environment.apiBaseUrl}/chat`);
    expect(req.request.body.question).toBe('What caused the loss?');
    req.flush({
      success: true,
      message: 'OK',
      data: {
        conversationId: 'conv-1',
        messageId: 'm1',
        answer: 'A burst pipe caused the loss.',
        sources: [{ fileName: 'claim.pdf', pageNumber: 2, relevanceScore: 0.9 }],
      },
    });
  });
});
