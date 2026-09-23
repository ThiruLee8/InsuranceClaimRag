import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { AgentLoopService } from './agent-loop.service';
import { environment } from '../../../environments/environment';

describe('AgentLoopService', () => {
  let service: AgentLoopService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(AgentLoopService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('loads agent-loop meta', () => {
    service.getMeta().subscribe((meta) => {
      expect(meta.tools.length).toBe(1);
      expect(meta.defaultMaxSteps).toBe(8);
    });

    const req = http.expectOne(`${environment.apiBaseUrl}/agent-loop/meta`);
    req.flush({
      success: true,
      message: 'OK',
      data: {
        tools: [{ name: 'search_documents', description: 'Search', parameters: '{}' }],
        sampleTasks: [],
        defaultMaxSteps: 8,
        defaultMaxLlmCalls: 10,
        defaultMaxSeconds: 90,
        defaultModel: 'llama3.2',
      },
    });
  });

  it('posts MCP rediscover', () => {
    service.rediscoverMcp().subscribe((mcp) => {
      expect(mcp.serverName).toBe('claims-docs');
      expect(mcp.tools.length).toBe(1);
    });

    const req = http.expectOne(`${environment.apiBaseUrl}/mcp/rediscover`);
    expect(req.request.method).toBe('POST');
    req.flush({
      success: true,
      message: 'OK',
      data: {
        enabled: true,
        transport: 'memory',
        serverName: 'claims-docs',
        connected: true,
        whereAiRuns: 'host',
        whereAiDoesNotRun: 'server',
        endpoint: '/mcp',
        stdio: 'python -m app.mcp',
        authRequired: true,
        tools: [{ name: 'list_documents', description: 'List', parameters: '{}' }],
        resources: [],
        prompts: [],
        handshake: [],
        roles: {},
      },
    });
  });
});
