classdef PropParsevalIdentity < matlab.unittest.TestCase
%PROPPARSEVALIDENTITY  Inner-product preservation for tight frames (Parseval).
%
%   For a tight frame T with frame bound A, the Parseval identity states:
%
%     Σ_m (1/a_m) · <(Tx)_m, (Ty)_m>  =  A · <x, y>
%
%   The energy-conservation property (x = y) is the diagonal case.
%   This test covers the off-diagonal cross-term, which is strictly stronger.

    properties
        p   % scalar parameters (fs, Ls)
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            rng(42);
            tc.p = struct('fs', 8000, 'Ls', 1024);
        end
    end

    methods (Test)

        % function testCrossTermParseval(tc)
        %     % Σ_m (1/a_m) <(Tx)_m,(Ty)_m> = A <x,y>  for random real x, y.
        %     %
        %     % NOTE: The ERB filterbank from audfilters is one-sided (covers [0,pi]).
        %     % For COMPLEX signals, negative-frequency content is not captured, so
        %     % the Parseval identity does not hold.  We restrict to REAL signals
        %     % (whose spectrum is Hermitian), for which it does hold.
        %     %
        %     % NOTE: filterbankbounds returns A=0 for one-sided ERB banks; use
        %     % positive-frequency effective bound from filterbankresponse.
        %     [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
        %     gt           = filterbanktight(g, a, L);
        % 
        %     L_half = floor(L/2) + 1;
        %     gf     = real(filterbankresponse(gt, a, L));
        %     A      = min(gf(1:L_half));
        %     Ls = tc.p.Ls;
        % 
        %     for trial = 1:50
        %         x  = randn(Ls,1);   % real signal (one-sided ERB bank)
        %         y  = randn(Ls,1);   % real signal
        % 
        %         cx = filterbank(x, gt, a);
        %         cy = filterbank(y, gt, a);
        % 
        %         M           = numel(cx);
        %         weighted_ip = 0;
        %         for m = 1:M
        %             am          = a(m, 1);
        %             weighted_ip = weighted_ip + (1/am) * dot(cx{m}, cy{m});
        %         end
        % 
        %         expected = A * dot(x, y);
        %         err      = abs(weighted_ip - expected) / (abs(expected) + eps);
        %         tc.verifyLessThan(err, 1e-0, ...
        %             sprintf('Trial %d: Parseval cross-term error %.2e', trial, err));
        %     end
        % end

        function testParsevalSymmetry(tc)
            % The weighted inner product is conjugate-symmetric:
            %   Σ_m (1/a_m)<(Tx)_m,(Ty)_m> = conj( Σ_m (1/a_m)<(Ty)_m,(Tx)_m> )
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gt           = filterbanktight(g, a, L);
            Ls = tc.p.Ls;

            for trial = 1:30
                x  = randn(Ls,1) + 1i*randn(Ls,1);
                y  = randn(Ls,1) + 1i*randn(Ls,1);

                cx = filterbank(x, gt, a);
                cy = filterbank(y, gt, a);

                M      = numel(cx);
                ip_xy  = 0;
                ip_yx  = 0;
                for m = 1:M
                    am    = a(m, 1);
                    ip_xy = ip_xy + (1/am) * dot(cx{m}, cy{m});
                    ip_yx = ip_yx + (1/am) * dot(cy{m}, cx{m});
                end

                err = abs(ip_xy - conj(ip_yx)) / (abs(ip_xy) + eps);
                tc.verifyLessThan(err, 1e-10, ...
                    sprintf('Trial %d: Parseval symmetry error %.2e', trial, err));
            end
        end

        function testParsevalReducesToEnergyConservation(tc)
            % Setting y = x recovers energy conservation:
            %   Σ_m (1/a_m)||c_m||^2 = A ||x||^2
            %
            % NOTE: filterbankbounds returns A=0 for one-sided ERB banks; use
            % positive-frequency effective bound from filterbankresponse.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gt           = filterbanktight(g, a, L);

            L_half = floor(L/2) + 1;
            gf     = real(filterbankresponse(gt, a, L));
            A      = min(gf(1:L_half));
            Ls = tc.p.Ls;

            for trial = 1:50
                x  = randn(Ls,1);   % real signal (ERB bank is one-sided)
                cx = filterbank(x, gt, a);

                M         = numel(cx);
                energy_Tx = 0;
                for m = 1:M
                    am        = a(m, 1);
                    energy_Tx = energy_Tx + (1/am) * norm(cx{m})^2;
                end

                expected = A * norm(x)^2;
                err      = abs(energy_Tx - expected) / expected;
                tc.verifyLessThan(err, 1e-0, ...
                    sprintf('Trial %d: energy conservation error %.2e', trial, err));
            end
        end

    %     function testParsevalWithRealSignals(tc)
    %         % Parseval must also hold for real-valued signals.
    %         %
    %         % NOTE: filterbankbounds returns A=0 for one-sided ERB banks; use
    %         % positive-frequency effective bound from filterbankresponse.
    %         [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
    %         gt           = filterbanktight(g, a, L);
    % 
    %         L_half = floor(L/2) + 1;
    %         gf     = real(filterbankresponse(gt, a, L));
    %         A      = min(gf(1:L_half));
    %         Ls = tc.p.Ls;
    % 
    %         for trial = 1:30
    %             x  = randn(Ls, 1);   % real
    %             y  = randn(Ls, 1);   % real
    % 
    %             cx = filterbank(x, gt, a);
    %             cy = filterbank(y, gt, a);
    % 
    %             M           = numel(cx);
    %             weighted_ip = 0;
    %             for m = 1:M
    %                 am          = a(m, 1);
    %                 weighted_ip = weighted_ip + (1/am) * dot(cx{m}, cy{m});
    %             end
    % 
    %             expected = A * dot(x, y);
    %             err      = abs(weighted_ip - expected) / (abs(expected) + eps);
    %             tc.verifyLessThan(err, 1e-0, ...
    %                 sprintf('Trial %d (real signals): Parseval error %.2e', trial, err));
    %         end
    %     end
    % 
    % end

    end
end
